import json
import os
import shutil
import yaml
import csv
from tqdm import tqdm
import pandas as pd
import argparse

VALID_FILES_PATH = "/home/aarav/nocturne_data/formatted_json_v2_no_tl_valid"
FINAL_BINNED_JSON_PATHS = "/home/aarav/nocturne_data/binned_jsons"
CSV_PATH = "/home/aarav/nocturne_data/datacsv.csv"

def modify_valid_files_json(valid_files_path: str, file_path: str):
    if(os.path.exists(valid_files_path + "/valid_files.json") == False):
        with open(valid_files_path + "/valid_files.json", 'w') as file:
            json.dump({}, file)
    with open(valid_files_path + "/valid_files.json", 'r') as file:
        valid_files = json.load(file)
    valid_files.clear()
    valid_files[file_path] = []
    with open(valid_files_path + "/valid_files.json", 'w') as file:
        json.dump(valid_files, file)

def delete_file_from_dest(file_path: str):
    os.remove(file_path)

def copy_file_to_dest(file_path: str, dest_path: str):
    shutil.copy(file_path, dest_path)
    return os.path.join(dest_path, os.path.basename(file_path))

def return_list_of_files(valid_files_path: str):
    with open(valid_files_path + "/valid_files.json", 'r') as file:
        valid_files = json.load(file)
    file_list = []
    for file in valid_files:
        file_list.append(os.path.join(valid_files_path, file))
    return file_list

def return_agent_numbers(file_path: str):
    with open(file_path, 'r') as file:
        data = json.load(file)
    num_agents = len(data['objects'])
    num_roads = len(data['roads'])
    num_road_segments = 0
    for road in data['roads']:
        if(road['type'] == "road_edge" or road['type'] == "road_line" or road['type'] == "lane"):
            num_road_segments += len(road['geometry']) - 1
        else:
            num_road_segments += 1
    return num_agents, num_road_segments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Nocturne Dataset Binning Tool')
    parser.add_argument('--files_path', type=str, help='Path to the valid files', default=VALID_FILES_PATH, required=False)
    parser.add_argument('--final_binned_json_paths', type=str, help='Path to the final binned json paths', default=FINAL_BINNED_JSON_PATHS, required=False)
    parser.add_argument('--csv_path', type=str, help='Path to the csv file', default=CSV_PATH, required=False)
    parser.add_argument('--bin_size', type=int, help='Num envs in a bin', default=100, required=False)
    args = parser.parse_args()

    VALID_FILES_PATH = args.files_path
    FINAL_BINNED_JSON_PATHS = args.final_binned_json_paths
    CSV_PATH = args.csv_path
    bin_size = args.bin_size

    file_list = return_list_of_files(VALID_FILES_PATH)
    file_meta_data = []
    file_meta_data.append(["File Path", "Number of Agents", "Number of Roads"])
    for file in tqdm(file_list):
        num_entities = return_agent_numbers(file)
        file_meta_data.append([file, num_entities[0], num_entities[1]])

    # Save bins for future use
    with open(CSV_PATH, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(file_meta_data)

    data = pd.read_csv(CSV_PATH)
    sorted_data = data.sort_values('Number of Agents')

    bins = []
    number_of_bins = len(sorted_data) // bin_size + (1 if len(sorted_data) % bin_size > 0 else 0)

    for i in range(number_of_bins):
        bin_start = i * bin_size
        bin_end = min((i + 1) * bin_size, len(sorted_data))
        bins.append(sorted_data.iloc[bin_start:bin_end])

    if not os.path.exists(FINAL_BINNED_JSON_PATHS):
        os.makedirs(FINAL_BINNED_JSON_PATHS)
    
    for i, bin in enumerate(bins):
        if not os.path.exists(FINAL_BINNED_JSON_PATHS + f"/bin_{i}"):
            os.makedirs(FINAL_BINNED_JSON_PATHS + f"/bin_{i}")
        bin_folder = FINAL_BINNED_JSON_PATHS + f"/bin_{i}"
        print(bin_folder)
        d = {}
        for index, row in bin.iterrows():
            file_path = row['File Path']
            d[file_path] = [row['Number of Agents'], row['Number of Roads']]
        filepath = os.path.join(bin_folder, f"valid_files.json")
        print(filepath)
        with open(filepath, 'w') as file:
            json.dump(d, file)
    print("Binning complete")