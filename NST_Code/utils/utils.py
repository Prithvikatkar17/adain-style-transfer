from torch.utils.data import Dataset
import os


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root))
        self.files = [p for p in self.files if p.endswith(('.jpg', '.jpeg', '.png'))]


    def __len__(self):
        return len(self.files)