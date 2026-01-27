import argparse
import pandas as pd
import sys
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate weighted average metrics from multiple JSONL files."
    )
    parser.add_argument(
        '--base_path', 
        type=str, 
        required=True, 
        help="Base directory path containing the input data files"
    )
    parser.add_argument(
        '--jsons', 
        type=str, 
        required=True, 
        help="Comma-separated list of filenames (e.g., file1.jsonl,file2.jsonl)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    file_list = [f.strip() for f in args.jsons.split(',')]
    output_file = os.path.join(args.base_path, 'WOSAC.csv')
    
    all_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, list):
                    all_results = loaded_data
        except (json.JSONDecodeError, ValueError):
            print(f"Warning: Could not read existing {output_file}. Creating a new one.")
            all_results = []

    for filename in file_list:
        file_path = os.path.join(args.base_path, filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: The file '{file_path}' does not exist. Skipping.")
            continue

        try:
            df = pd.read_json(file_path, lines=True)
            print(f"[{filename}] Successfully loaded {len(df)} rows.")

        except ValueError as e:
            print(f"[{filename}] Error reading file: {e}")
            continue

        weight_col = 'total_num_agents'
        
        if weight_col not in df.columns:
            print(f"[{filename}] Error: Column '{weight_col}' not found. Skipping.")
            continue

        numeric_cols = df.select_dtypes(include=['number']).columns
        metric_cols = [col for col in numeric_cols if col != weight_col]

        total_weight_sum = df[weight_col].sum()
        
        file_result = {
            "filename": filename,
            "total_agents_sum": int(total_weight_sum)
        }

        for metric in metric_cols:
            weighted_sum = (df[metric] * df[weight_col]).sum()
            weighted_avg = weighted_sum / total_weight_sum
            file_result[metric] = float(weighted_avg)

        all_results.append(file_result)
        
        print(f"\n{'='*20} Results for: {filename} {'='*20}")
        print(f"Total Agents Sum (Denominator): {int(total_weight_sum)}")
        print("-" * 65)
        print(f"{'Metric Name':<45} | {'Weighted Avg':<15}")
        print("-" * 65)
        
        for key in sorted(file_result.keys()):
            if key not in ["filename", "total_agents_sum"]:
                print(f"{key:<45} | {file_result[key]:.6f}")
        print("-" * 65)
        
        print(f"[{filename}] Processed and appended metrics.")

    try:
        if all_results:
            final_df = pd.DataFrame(all_results)
            
            cols = ['filename', 'total_agents_sum'] + [c for c in final_df.columns if c not in ['filename', 'total_agents_sum']]
            final_df = final_df[cols]
            
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print("-" * 50)
            print(f"All done! Results saved to '{output_file}'")
        else:
            print("No results to save.")

    except Exception as e:
        print(f"Error saving to {output_file}: {e}")

if __name__ == "__main__":
    main()