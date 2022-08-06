import os
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MyData(Dataset):
    def __init__(self,dir):
        self.dir = dir
        self.transform = transforms.Compose([
            transforms.Resize([32,32]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        for root,dirs,files in os.walk(self.dir):
            self.files = files

    def __getitem__(self,idx):
        file = self.files[idx]
        cursor = int(file.split('.')[0].split("_")[1])
        self.img = Image.open(self.dir + file)
        self.img = self.transform(self.img)
        return self.img,cursor

    def __len__(self):
        return len(self.files)
