import argparse
from altair import value
import torch
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--content_dir',
        type=str,
        default=r'NST_Code/content_data',
        help='Location of content dataset'
    )

    parser.add_argument(
        '--style_dir',
        type=str,
        default=r'NST_Code/style_data',
        help='Location of style dataset'
    )

    parser.add_argument(
        '--vgg',
        type=str,
        default='',
        help='Location of pretrained VGG'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        default='experiment',
        help='Name of experiment'
    )

    return parser.parse_args()

def main():
    args = parse_arguments()
    print(args.experiment)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = Path('experiments') / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)


    # save arguments valvues to a text file
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')
            

if __name__ == "__main__":
    main()