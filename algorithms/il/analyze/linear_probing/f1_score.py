from sklearn.metrics import f1_score
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--')
    parser.add_argument('--model-path', '-en', type=str, default='all_data')
    parser.add_argument('--sweep-id', type=str, default=None)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--use-mask', action='store_true')
    parser.add_argument('--use-tom', '-ut', default=None, choices=[None, 'guide_weighted', 'no_guide_no_weighted',
                                                                   'no_guide_weighted', 'guide_no_weighted'])
    parser.add_argument('--ego-future-step', '-afs', type=int, default=30)
    parser.add_argument('--baseline', '-b', action='store_true')    
    args = parser.parse_args()
    
    return args

def main(args):
    models_path = f'/data/model/{args.exp_name}/linear_prob/{args.model_name}'
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
