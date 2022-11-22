
from torch.utils.data import Dataset
import glob
import numpy as np
import cv2
import os
import torch

dataset_directory = "dataset/fruits-360_dataset/fruits-360/divided_training/"


class CustomDataset(Dataset):
    def __init__(self,transform=None):
        self.transform = transform
        self.imgs_path = dataset_directory
        file_list = glob.glob(self.imgs_path + "*")
        # print(file_list)
        count = 0
        self.data = []
        for macrocategory in os.listdir(self.imgs_path):
            for subcategory in os.listdir(self.imgs_path+"/"+macrocategory):
                for filename in os.listdir(self.imgs_path+"/"+macrocategory+'/'+subcategory):
                    img = cv2.imread(os.path.join(self.imgs_path+"/"+macrocategory+'/'+subcategory,filename))
                    if img is not None:
                        img = cv2.resize(img, (45,45), interpolation = cv2.INTER_AREA)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = np.array(img).swapaxes(0,2).swapaxes(1,2)
                        # self.data.append([np.expand_dims(img[0], axis=0),count])
                        self.data.append([img,count])
            count+=1
        # print(self.data)
        # self.data = np.array(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, class_id = self.data[idx]
        img_tensor = torch.tensor(np.array(img),dtype=torch.float32)
        class_id = torch.tensor(np.array(class_id))
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor/255, class_id