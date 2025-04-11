import argparse
import subprocess
from tqdm import tqdm


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_idx", "-g", default=0)
    
    parser.add_argument("--file_num", "-fn", type=int, default=134400, help="Train : 134400, Valid : 12201")
    parser.add_argument("--dataset", "-d", choices=["train", "valid"])
    parser.add_argument("--num_worlds", "-nw", type=int, default=200)
    parser.add_argument("--start_idx", "-si", type=int, default=0)
    parser.add_argument("--save_dir", "-sd", type=str, default="/root/pygpudrive/pygpudrive/data/media")
    parser.add_argument("--function", "-f", default="save_video", choices=["save_video", "save_frame"])
    
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    gpu_index = args.gpu_idx    
    file_num = args.file_num
    dataset = args.dataset
    num_worlds = args.num_worlds
    start_idx = args.start_idx
    function = args.function
    
    for i in tqdm(range(start_idx, start_idx + file_num // num_worlds)):
        save_index = start_idx + num_worlds * (i - start_idx)
        command = f"CUDA_VISIBLE_DEVICES={gpu_index} /root/anaconda3/envs/gpudrive/bin/python pygpudrive/utils/save_media.py --dataset {dataset} --num_worlds {num_worlds} --start_idx {save_index} --function {function}"    
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f"Error: Command failed with return code {result.returncode}")