import argparse
import json
import os
import numpy as np

def generate_indices(data_path, num_input_views, num_target_views, output_dir):
    base_dir = os.path.dirname(data_path)
    data_point_paths = json.load(open(data_path, "r"))
    
    indices_map = {}
    # We need K = Ni + Nt total views
    num_total_needed = num_input_views + num_target_views
    
    print(f"Generating indices for {len(data_point_paths)} scenes.")
    print(f"Config: Input={num_input_views}, Target={num_target_views}, Total={num_total_needed}")

    for path in data_point_paths:
        full_path = os.path.join(base_dir, path)
        if not os.path.exists(full_path): 
            print(f"Warning: {full_path} not found, skipping.")
            continue
        
        scene_name = os.path.basename(os.path.dirname(full_path))
        try:
            with open(full_path, "r") as f:
                info = json.load(f)
                num_available = len(info)
        except Exception as e:
            print(f"Error reading {full_path}: {e}")
            continue

        if num_available == 0: continue

        # Select indices uniformly from available frames
        selected_indices = np.linspace(0, num_available - 1, num_total_needed, dtype=int)
        
        # Distribute into Input/Target buckets to interleave spatially
        # Inputs get even positions in the sorted selected list
        # Targets get odd positions
        
        input_bucket = []
        target_bucket = []
        
        for i in range(num_total_needed):
            idx = int(selected_indices[i])
            if i % 2 == 0:
                if len(input_bucket) < num_input_views:
                    input_bucket.append(idx)
                else:
                    target_bucket.append(idx)
            else:
                if len(target_bucket) < num_target_views:
                    target_bucket.append(idx)
                else:
                    input_bucket.append(idx)
        
        final_indices = input_bucket + target_bucket
        indices_map[scene_name] = final_indices

    output_filename = f"test_indices_in{num_input_views}_tar{num_target_views}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(indices_map, f, indent=4)
    
    print(f"Saved test indices to {output_path}")

if __name__ == "__main__":
    # python generate_indices.py --data_path data_example/dl3dv_10k_sample_data_path.json --num_input_views 32 --num_target_views 32 --output_dir data_example/
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data index json file")
    parser.add_argument("--num_input_views", type=int, default=32)
    parser.add_argument("--num_target_views", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    
    generate_indices(args.data_path, args.num_input_views, args.num_target_views, args.output_dir)

