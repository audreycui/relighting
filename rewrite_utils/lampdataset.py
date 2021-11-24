import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import PIL.Image as Image
import numpy as np

class DarkenedLampsDataset(Dataset):
    # need to define your __init__()
    def __init__(self): 
        super(DarkenedLampsDataset, self).__init__()
        self.data_root = '/data/vision/torralba/ganlight/datasets/darkened_lamps/original'
        self.img_list = glob.glob(os.path.join(self.data_root,'*.png'))
        self.data_length = len(self.img_list)
        
    # need to define your __len__()
    def __len__(self):
        return self.data_length
    
    # need to define your __get_item__()
    def __getitem__(self, idx):
        im_name = self.img_list[idx]
        image = Image.open(im_name)
        # need 1) make it [-1,1] 2) HWC [100,100,3] ==> CHW [3, 100, 100], send me np (or tensor)
        image = np.array(image)
        image = (image/255 - 0.5) * 2
        image = np.transpose(image, (2,0,1))
        image = torch.tensor(image, dtype=torch.float32)
        # same way load z_vect and other stuff
        z_name = im_name.replace('original_', 'z_').replace('original', 'npy').replace('png', 'npy')
        z_vect = np.load(z_name)
        z_vect = torch.tensor(z_vect, dtype=torch.float32).squeeze()

#         mask_name = im_name.replace('original_', 'mask_').replace('original', 'mask')
#         mask = Image.open(mask_name)
#         mask = np.array(mask)
#         mask = mask/255 #should be in [0, 1]
#         mask = torch.tensor(mask, dtype=torch.float32)
        
        target_name = im_name.replace('original_', 'new_target_').replace('original', 'target')
        target = Image.open(target_name)
        # need 1) make it [-1,1] 2) HWC [100,100,3] ==> CHW [3, 100, 100], send me np (or tensor)
        target = np.array(target)
        target = (target/255 - 0.5) * 2
        target = np.transpose(target, (2,0,1))
        target = torch.tensor(target, dtype=torch.float32)

        seg_name = im_name.replace('original_', 'segmented_npy_').replace('original', 'segmented_npy').replace('png', 'npy')
        segment = torch.tensor(np.load(seg_name, allow_pickle=True), dtype=torch.float32)

        return [z_vect, image, target, segment]
#        return [z_vect, image, mask, target]
