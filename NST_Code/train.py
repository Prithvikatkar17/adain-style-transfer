import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir' , type = str ,default='NST_Code\content_data',
                        help='Location of content dataset')
    parser.add_argument('--style_dir' , type= str , default='NST_Code\style_data',
                        help='Location of style dataset')
    parser.add_argument('--vgg' ,type=str , default='',
                        help='Location of pretrained VGG')
    parser.add_argument('--exepriment' ,type=str , default='exepriment' ,
                        help='Name of exeriment')

    return parser.parse_args() 

def main():
    args = parse_arguments()
    print(args.exepriment)

if __name__ == "__main__":
    main()