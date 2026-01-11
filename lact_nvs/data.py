import json
import os
import random
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def resize_and_crop(image, target_size, fxfycxcy):
    """
    Resize and crop image to target_size, adjusting camera parameters accordingly.
    
    Args:
        image: PIL Image
        target_size: (height, width) tuple
        fxfycxcy: [fx, fy, cx, cy] list
    
    Returns:
        tuple: (resized_cropped_image, adjusted_fxfycxcy)
    """
    original_width, original_height = image.size  # PIL image is (width, height)
    target_height, target_width = target_size
    
    fx, fy, cx, cy = fxfycxcy
    
    # Calculate scale factor to fill target size (resize to cover)
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale = max(scale_x, scale_y)  # Use larger scale to ensure it covers the target area
    
    # Resize image
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate crop box for center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    # Crop image
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    # Adjust camera parameters
    # Scale focal lengths and principal points
    new_fx = fx * scale
    new_fy = fy * scale
    new_cx = cx * scale - left
    new_cy = cy * scale - top
    
    return cropped_image, [new_fx, new_fy, new_cx, new_cy]


def normalize(x):
    return x / x.norm()

def normalize_with_mean_pose(c2ws: torch.Tensor):
    # This is a historical code for scene camera normalization;
    #  thanks to the authors (might mostly credit to Zexiang Xu)

    # Get the mean parameters
    center = c2ws[:, :3, 3].mean(0)
    vec2 = c2ws[:, :3, 2].mean(0)
    up = c2ws[:, :3, 1].mean(0)

    # Get the view matrix.
    vec2 = normalize(vec2)
    vec0 = normalize(torch.cross(up, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, center], 1)

    # Extend the view matrix to 4x4.
    avg_pos = c2ws.new_zeros(4, 4)
    avg_pos[3, 3] = 1.0
    avg_pos[:3] = m

    # Align coordinate system to the mean camera
    c2ws = torch.linalg.inv(avg_pos) @ c2ws

    # Scale the scene to the range of [-1, 1].
    scene_scale = torch.max(torch.abs(c2ws[:, :3, 3]))
    c2ws[:, :3, 3] /= scene_scale

    return c2ws


class NVSDataset(Dataset):
    def __init__(self, 
        data_path, num_views, image_size, 
        sorted_indices=False, 
        scene_pose_normalize=False,
        fixed_indices=None,
        fdist_min=None,
        fdist_max=None,
    ):
        """
        image_size is (h, w) or just a int (as size).
        fixed_indices: optional dict {scene_name: [indices...]} to override selection logic.
        fdist_min: minimum frame index (None = start of video, i.e. 0)
        fdist_max: maximum frame index (None = end of video, i.e. total_frames)
        """
        self.base_dir = os.path.dirname(data_path)
        self.data_point_paths = json.load(open(data_path, "r"))
        self.sorted_indices = sorted_indices
        self.scene_pose_normalize = scene_pose_normalize
        self.fixed_indices = fixed_indices
        self.fdist_min = fdist_min
        self.fdist_max = fdist_max
        print(f"fdist_min: {fdist_min}, fdist_max: {fdist_max}")

        # filter out the scenes that have less than num_views images
        original_num_scenes = len(self.data_point_paths)
        self.data_point_paths = [path for path in self.data_point_paths if len(json.load(open(os.path.join(self.base_dir, path), "r"))) >= num_views]
        filtered_num_scenes = len(self.data_point_paths)
        print(f"Found {original_num_scenes} scenes in the index file, filtered out {original_num_scenes - filtered_num_scenes} scenes with less than {num_views} images, remaining {filtered_num_scenes} scenes")

        # except for rendering video, at both training and inference time we always don't need to sort the indices
        assert not sorted_indices, "Except for rendering video, at both training and inference time we always don't need to sort the indices"

        self.num_views = num_views
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

    def __len__(self):
        return len(self.data_point_paths)
    
    def __getitem__(self, index):
        data_point_path = os.path.join(self.base_dir, self.data_point_paths[index])
        data_point_base_dir = os.path.dirname(data_point_path)
        scene_name = os.path.basename(data_point_base_dir)

        with open(data_point_path, "r") as f:
            images_info = json.load(f)
        
        # Determine indices
        if self.fixed_indices is not None:
            # assert scene_name in self.fixed_indices, f"Scene {scene_name} not found in fixed indices"
            # Use fixed indices provided externally
            # indices = self.fixed_indices[scene_name]

            # use Long-LRM input indices
            input_indices = self.fixed_indices[index][f"fold_8_kmeans_{((self.num_views + 1) // 2)}_input"]
            
            # Uniformly select target indices from the all views
            num_targets = (self.num_views + 1) // 2
            total_views = len(images_info)
            step = total_views // num_targets
            target_indices = [int(i * step + step // 2) for i in range(num_targets)]
            indices = input_indices + target_indices

            # If fixed_indices are used, we assume they are pre-ordered as desired (e.g. interleaved).
            # We do NOT sort them unless sorted_indices is explicitly True, but typically we turn it off.
            if self.sorted_indices:
                 indices = sorted(indices)
        else:
            # If the num_views is larger than the number of images, use all images
            # Sample from a subsection of the video based on fdist_min and fdist_max
            total_frames = len(images_info)
            fdist_min = self.fdist_min if self.fdist_min is not None else 0
            fdist_max = self.fdist_max if self.fdist_max is not None else total_frames
            # Ensure fdist is at least num_views so we can sample enough frames
            fdist_min = max(fdist_min, self.num_views)
            fdist_max = max(fdist_max, self.num_views)
            fdist = random.randint(fdist_min, fdist_max)
            if total_frames < fdist:
                return self.__getitem__((index + 1) % len(self.data_point_paths))
            start_idx = random.randint(0, total_frames - fdist)
            end_idx = start_idx + fdist
            indices = random.sample(range(start_idx, end_idx), self.num_views)
            if self.sorted_indices:
                indices = sorted(indices)
        
        fxfycxcy_list = []
        c2w_list = []
        image_list = []
        
        for index in indices:
            info = images_info[index]
            
            fxfycxcy = [info["fx"], info["fy"], info["cx"], info["cy"]]
            
            w2c = torch.tensor(info["w2c"])
            c2w = torch.inverse(w2c)
            c2w_list.append(c2w)
            
            # Load image from file_path using PIL and convert to torch tensor
            image_path = os.path.join(data_point_base_dir, info["file_path"])
            image = Image.open(image_path)
            
            image, fxfycxcy = resize_and_crop(image, self.image_size, fxfycxcy)

            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                # Create a white background and paste the RGBA image on it
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = rgb_image
            elif image.mode != 'RGB':
                # Convert any other mode to RGB
                image = image.convert('RGB')
            
            fxfycxcy_list.append(fxfycxcy)
            image_list.append(transforms.ToTensor()(image))
        
        c2ws = torch.stack(c2w_list)
        if self.scene_pose_normalize:
            # print("Normalizing scene poses...")
            c2ws = normalize_with_mean_pose(c2ws)

        return {
            "fxfycxcy": torch.tensor(fxfycxcy_list),
            "c2w": c2ws,
            "image": torch.stack(image_list),
            "indices": torch.tensor(indices),
            "scene_name": scene_name,
        }


def preprocess_poses_re10k(c2ws: torch.Tensor, scene_scale_factor: float = 1.35):
    """
    Preprocess the poses for Re10k dataset (per-batch normalization).
    1. Translate and rotate the scene to align the average camera direction and position
    2. Rescale the whole scene to a fixed scale
    """
    import torch.nn.functional as F
    
    # Translation and Rotation
    # align coordinate system (OpenCV coordinate) to the mean camera
    center = c2ws[:, :3, 3].mean(0)
    avg_forward = F.normalize(c2ws[:, :3, 2].mean(0), dim=-1)  # average forward direction (z of opencv camera)
    avg_down = c2ws[:, :3, 1].mean(0)  # average down direction (y of opencv camera)
    avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)  # (x of opencv camera)
    avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)  # (y of opencv camera)

    avg_pose = torch.eye(4, device=c2ws.device)  # average c2w matrix
    avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
    avg_pose[:3, 3] = center
    avg_pose = torch.linalg.inv(avg_pose)  # average w2c matrix
    c2ws = avg_pose @ c2ws

    # Rescale the whole scene to a fixed scale
    scene_scale = torch.max(torch.abs(c2ws[:, :3, 3]))
    scene_scale = scene_scale_factor * scene_scale
    c2ws[:, :3, 3] /= scene_scale

    return c2ws


class Re10kNVSDataset(Dataset):
    """
    Dataset for reading Re10k format data (scene_info.json with frames containing
    image_path, intrinsics, c2ws). Supports both directory and zip file reading.
    
    Uses LVSM-style view selection with num_input_views and num_target_views.
    Outputs the same format as NVSDataset for compatibility.
    """
    
    def __init__(self, 
        data_path, 
        num_input_views,
        num_target_views,
        image_size, 
        inference=False,
        scene_pose_normalize=True,  # Re10k uses per-batch normalization by default
        min_frame_dist=25,
        max_frame_dist=192,
        target_has_input=True,  # Whether target views can overlap with input views
        eval_index_path="data_example/evaluation_index_re10k.json",  # Default path for Re10k evaluation indices
    ):
        """
        Args:
            data_path: Path to directory or zip file containing scene folders
            num_input_views: Number of input views
            num_target_views: Number of target views
            image_size: (h, w) tuple or int for square size
            inference: Whether in inference mode (uses fixed indices from eval_index_path)
            scene_pose_normalize: Whether to normalize scene poses (per-batch normalization)
            min_frame_dist: Minimum frame distance for view selection
            max_frame_dist: Maximum frame distance for view selection
            target_has_input: Whether target views can overlap with input views (default True)
            eval_index_path: Path to evaluation index file for inference mode
        """
        self.num_input_views = num_input_views
        self.num_target_views = num_target_views
        self.inference = inference
        self.scene_pose_normalize = scene_pose_normalize
        self.min_frame_dist = min_frame_dist
        self.max_frame_dist = max_frame_dist
        self.target_has_input = target_has_input
        print(f"num_input_views: {num_input_views}, num_target_views: {num_target_views}")
        print(f"min_frame_dist: {min_frame_dist}, max_frame_dist: {max_frame_dist}, target_has_input: {target_has_input}")

        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        
        # Check if data_path is a zip file
        self.is_zip = data_path.endswith('.zip') and os.path.isfile(data_path)
        self.zip_path = data_path if self.is_zip else None
        self.dataset_path = data_path
        
        # List all scenes in the dataset
        if self.is_zip:
            print(f"Loading zip file: {data_path}")
            with zipfile.ZipFile(data_path, 'r') as zf:
                all_files = zf.namelist()
                scene_dirs = set()
                for file_path in all_files:
                    if 'scene_info.json' in file_path:
                        scene_dir = os.path.dirname(file_path)
                        scene_dirs.add(scene_dir)
                all_scene_paths = sorted(list(scene_dirs))
        else:
            scenes = os.listdir(data_path)
            all_scene_paths = [os.path.join(data_path, scene) for scene in scenes 
                               if os.path.isdir(os.path.join(data_path, scene))]
            all_scene_paths = sorted(all_scene_paths)
        
        # Load evaluation indices for inference mode
        if self.inference:
            assert os.path.exists(eval_index_path), \
                f"Evaluation index file {eval_index_path} does not exist."
            with open(eval_index_path, 'r') as f:
                view_idx_list = json.load(f)
            
            # Filter out None values and scenes that don't have specified input/target views
            view_idx_list_filtered = [k for k, v in view_idx_list.items() if v is not None]
            filtered_scene_paths = []
            for scene in all_scene_paths:
                if self.is_zip:
                    scene_name = os.path.basename(scene)
                else:
                    scene_name = scene.split("/")[-1]
                if scene_name in view_idx_list_filtered:
                    filtered_scene_paths.append(scene)
            
            print(f"Found {len(view_idx_list_filtered)} scenes in index file, {len(filtered_scene_paths)} scenes exist in the dataset.")
            all_scene_paths = filtered_scene_paths
            
            # Store indices as numpy arrays to prevent memory leaking
            input_idx_list_np, target_idx_list_np = [], []
            for scene_path in all_scene_paths:
                scene_name = os.path.basename(scene_path)
                assert scene_name in view_idx_list, f"Scene {scene_name} is not in the view idx list."
                input_idx_list_np.append(view_idx_list[scene_name]["context"])
                target_idx_list_np.append(view_idx_list[scene_name]["target"])
            self.input_idx_list_np = np.array(input_idx_list_np).astype(np.int32)
            self.target_idx_list_np = np.array(target_idx_list_np).astype(np.int32)
        
        print(f"Using {len(all_scene_paths)} scenes")
        
        # Store as numpy bytes array to prevent memory leaking
        self.all_scene_paths = np.array(all_scene_paths).astype(np.bytes_)
        
        # For zip files, extract the base directory name
        if self.is_zip and len(all_scene_paths) > 0:
            first_scene = all_scene_paths[0]
            parts = first_scene.split('/')
            if len(parts) > 1:
                self.zip_base_dir = parts[0]
            else:
                self.zip_base_dir = ""
        else:
            self.zip_base_dir = None
        
        # ZipFile object for worker processes (initialized lazily)
        self._zip_file = None
        self._worker_id = None

    def __len__(self):
        return len(self.all_scene_paths)
    
    def _get_zip_file(self):
        """Get or create a ZipFile object for the current worker process."""
        if not self.is_zip:
            return None
        
        import torch.utils.data
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        
        if self._zip_file is None or self._worker_id != worker_id:
            if self._zip_file is not None:
                self._zip_file.close()
            self._zip_file = zipfile.ZipFile(self.zip_path, 'r')
            self._worker_id = worker_id
        
        return self._zip_file
    
    def _load_image(self, image_path, scene_path):
        """Load image from disk or zip file."""
        if self.is_zip:
            zf = self._get_zip_file()
            # image_path from JSON is relative (e.g., "scene_name/00000.png")
            # zip file stores with base directory prefix (e.g., "test/scene_name/00000.png")
            if self.zip_base_dir:
                zip_image_path = f"{self.zip_base_dir}/{image_path}"
            else:
                zip_image_path = image_path
            with zf.open(zip_image_path) as f:
                image = Image.open(f)
                image = image.copy()  # Load into memory before closing file handle
                return image
        else:
            abs_image_path = os.path.join(self.dataset_path, image_path)
            return Image.open(abs_image_path)
    
    def _view_selector(self, frames):
        """Select input and target view indices following LVSM logic."""
        num_total = self.num_input_views + self.num_target_views
        if len(frames) < num_total:
            return None
        
        if self.max_frame_dist <= self.min_frame_dist:
            return None
        
        # Distance between input views
        frame_dist = random.randint(self.min_frame_dist, self.max_frame_dist)
        if len(frames) <= frame_dist:
            return None
        
        start_frame = random.randint(0, len(frames) - frame_dist - 1)
        end_frame = start_frame + frame_dist
        
        # Input views are the start and end frames (like LVSM)
        input_indices = [start_frame, end_frame]
        
        # If we need more than 2 input views, sample additional ones
        if self.num_input_views > 2:
            additional_input = random.sample(
                range(start_frame + 1, end_frame), 
                self.num_input_views - 2
            )
            input_indices.extend(additional_input)
        
        # Target views selection
        if self.target_has_input:
            # Target views can overlap with input views - sample from all frames in range
            available_for_target = list(range(start_frame, end_frame + 1))
        else:
            # Target views must be different from input views
            available_for_target = [i for i in range(start_frame, end_frame + 1) 
                                    if i not in input_indices]
        
        if len(available_for_target) < self.num_target_views:
            return None
        
        target_indices = random.sample(available_for_target, self.num_target_views)
        
        # Return input indices first, then target indices
        return input_indices + target_indices
    
    def __getitem__(self, index):
        scene_path = str(self.all_scene_paths[index], encoding="utf-8").strip()
        json_file_path = os.path.join(scene_path, "scene_info.json")
        
        try:
            if self.is_zip:
                zf = self._get_zip_file()
                with zf.open(json_file_path) as f:
                    data_json = json.load(f)
            else:
                with open(json_file_path, 'r') as f:
                    data_json = json.load(f)
        except Exception as e:
            # Handle IO errors by reading next scene
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        frames = data_json["frames"]
        scene_name = data_json["scene_name"]
        
        # Determine indices
        if self.inference:
            input_indices = list(self.input_idx_list_np[index])
            target_indices = list(self.target_idx_list_np[index])
            assert self.num_input_views == len(input_indices), \
                f"Expected {self.num_input_views} input views, got {len(input_indices)}"
            assert self.num_target_views == len(target_indices), \
                f"Expected {self.num_target_views} target views, got {len(target_indices)}"
            indices = input_indices + target_indices
        else:
            # Sample input and target views using LVSM-style view selector
            indices = self._view_selector(frames)
            if indices is None:
                return self.__getitem__(random.randint(0, len(self) - 1))
        
        fxfycxcy_list = []
        c2w_list = []
        image_list = []
        
        for idx in indices:
            frame = frames[idx]
            
            # Re10k format: intrinsics is [fx, fy, cx, cy]
            fxfycxcy = list(frame["intrinsics"])
            
            # Re10k format: c2ws is stored directly (not w2c)
            c2w = torch.tensor(frame["c2ws"]).float()
            c2w_list.append(c2w)
            
            # Load image
            try:
                image = self._load_image(frame["image_path"], scene_path)
            except Exception as e:
                return self.__getitem__(random.randint(0, len(self) - 1))
            
            # Resize and crop to target size
            image, fxfycxcy = resize_and_crop(image, self.image_size, fxfycxcy)
            
            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])
                image = rgb_image
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            fxfycxcy_list.append(fxfycxcy)
            image_list.append(transforms.ToTensor()(image))
        
        c2ws = torch.stack(c2w_list)
        if self.scene_pose_normalize:
            # Use LVSM-style per-batch normalization
            c2ws = preprocess_poses_re10k(c2ws)
        
        return {
            "fxfycxcy": torch.tensor(fxfycxcy_list),
            "c2w": c2ws,
            "image": torch.stack(image_list),
            "indices": torch.tensor(indices),
            "scene_name": scene_name,
        }