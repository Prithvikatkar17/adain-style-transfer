from matplotlib import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root))
        self.files = [p for p in self.files if p.endswith(('.jpg', '.jpeg', '.png'))]


    def __len__(self):
        return len(self.files)

    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)


        return image
    
def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))
        
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)