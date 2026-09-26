import argparse
from hpack import Decoder
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from NST_Code.utils.models import VGGEncoder
from utils.utils import *

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

    parser.add_argument(
        '--final_size',
        type=int,
        default=256,
        help='Final size of the image'
    )

    parser.add_argument(
        '--content_size',
        type=int,
        default=512,
        help='Size of the content image'
    )

    parser.add_argument(
        '--style_size',
        type=int,
        default=512,
        help='Size of the style image'
    )

    parser.add_argument(
        '--crop',
        action='store_true',
        default=True,
        help='crop images'
    )
    parser.add_argument('--batch_size', type=int, default=4)
    return parser.parse_args()

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')  

    parser.add_argument('--lr_decay', type=float, default=5e-5, help='Learning rate decay factor for the scheduler')

def main():
    args = parse_arguments()
    print(args.experiment)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = Path('experiments') / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)


    # save arguments values to a text file
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')

    content_transform = get_transform(args.content_size, args.crop , args.final_size)
    style_transform = get_transform(args.style_size, args.crop , args.final_size)


    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_loader = DataLoader(content_dataset,
                                batch_size= args.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)

    style_loader = DataLoader(style_dataset,
                                batch_size= args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)


    print("number of content batches in content dataset :",len(content_dataset))
    print("number of style batches in style dataset :" , len(style_dataset))

    for batch in style_loader :
        print(batch.shape)


    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)


    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          lr_lambda=lambda epoch: 1.0 / (1.0 +args.lr_decay * epoch))

        
            

if __name__ == "__main__":
    main()