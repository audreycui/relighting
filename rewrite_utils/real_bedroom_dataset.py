import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import PIL.Image as Image
import numpy as np

class RealBedroomDataset(Dataset):
    # need to define your __init__()
    def __init__(self): 
        super(RealBedroomDataset, self).__init__()
        self.data_root = '/data/vision/torralba/datasets/LSUN/image/bedroom_val'
        self.img_list = glob.glob(os.path.join(self.data_root,'*.webp'))
        self.data_length = len(self.img_list)
        
    # need to define your __len__()
    def __len__(self):
        return self.data_length
    
    # need to define your __get_item__()
    def __getitem__(self, idx):
        im_name = self.img_list[idx]
        image = Image.open(im_name)
        # need 1) make it [-1,1] 2) HWC [100,100,3] ==> CHW [3, 100, 100], send me np (or tensor)
        image = image.crop((0, 0, 256, 256)) 
        image = np.array(image)
        image = (image/255 - 0.5) * 2
        image = np.transpose(image, (2,0,1))
        image = torch.tensor(image, dtype=torch.float32)
        

        return [image]
#        return [z_vect, image, mask, target]
