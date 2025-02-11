"""Extract expert states and actions from Waymo Open Dataset."""
import logging
import subprocess
import argparse
import os
from tqdm import tqdm

logging.getLogger(__name__)

def arg_parse():
    parser = argparse.ArgumentParser()
    # MODEL
    parser.add_argument('--model-path', '-mp', type=str, default='/data/model/eat100to1000')
    # GPU SETTINGS
    parser.add_argument('--gpu-id', '-g', type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    
    models = [os.path.splitext(model)[0] for model in os.listdir(args.model_path) if model.endswith('.pth')]
    
    for model in tqdm(models):
        for dataset in ['train', 'valid']:
            print("model: ", model)
            
            command = f"CUDA_VISIBLE_DEVICES={args.gpu_id} /root/anaconda3/envs/gpudrive/bin/python algorithms/il/evaluate.py -mc -mv -m {model} --dataset {dataset}"
            
            result = subprocess.run(command, shell=True)
            
            if result.returncode != 0:
                print(f"Error: Command failed with return code {result.returncode}")
