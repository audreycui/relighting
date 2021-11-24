from torch.utils.data import Dataset, DataLoader
from data.base_data_loader import BaseDataLoader
from models.masked_stylegan import make_masked_stylegan
from torch.utils.data.sampler import WeightedRandomSampler

import torch 
import torch.nn.functional as F

import os, sys, inspect
from os import path

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from rewrite_utils import zdataset, show, labwidget, paintwidget, renormalize, nethook, imgviz, pbar, smoothing, imgviz
from rewrite_utils.stylegan2 import load_seq_stylegan
import copy, contextlib
from rewrite_utils.stylegan2.models import DataBag
from rewrite_utils.segmenter import load_segmenter
from importlib import reload

from PIL import Image
import random
import numpy as np

from torchvision.transforms import Resize
import cv2


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

class StyleGANDatasetDataLoader(BaseDataLoader): 
    def name(self): 
        return 'StyleGANDatasetDataLoader'
    
    def initialize(self, opt): 
        BaseDataLoader.initialize(self, opt)
        if opt.masked: 
            on_dataset = MaskedStyleGANDataset()
            off_dataset = MaskedStyleGANDataset(reverse = True)
            self.dataset = torch.utils.data.ConcatDataset([on_dataset, off_dataset])
        elif opt.alternate_train: 
            on_dataset = AlternateStyleGANDataset()
            off_dataset = AlternateStyleGANDataset(reverse = True)
            self.dataset = torch.utils.data.ConcatDataset([on_dataset, off_dataset])
 
        else: 
            if opt.n_stylechannels > 1:
                on_dataset = MultichannelStyleGANDataset(2)
                off_dataset = MultichannelStyleGANDataset(2, reverse = True)
                self.dataset = torch.utils.data.ConcatDataset([on_dataset, off_dataset])
            else:
                if opt.isTrain: 
                    loc_map = opt.use_location_map
                    bootstrap = opt.lamp_off_bootstrap
                    self.dataset = StyleGANDataset(loc_map=loc_map, bootstrap=bootstrap)
                else: 
                    self.dataset = StyleGANDataset(frac_one = opt.frac_one)
        self.dataloader = DataLoader( 
            self.dataset, 
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            #num_workers=int(opt.nThreads))
            num_workers=0)
        
    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
        
class StyleGANDataset(Dataset): 
    def __init__(self, 
                 loc_map=False, 
                 bootstrap=False,
                 dset='bedroom', 
                 debug = False, 
                 frac_one = False, 
                ): 
        self.model = load_seq_stylegan('bedroom', mconv='seq', truncation=0.90)
        nethook.set_requires_grad(False, self.model)
        self.light_layer = 'layer8'
        self.light_unit = 265
        #self.segmodel, self.seglabels = load_segmenter()
        self.color = torch.tensor([1.0, 1.0, 1.0]).float().cuda()[:,None,None]
        self.num = 0
        self.use_loc = loc_map
        self.bootstrap = bootstrap
        self.frac_one = frac_one
        if self.use_loc: 
            self.loc_frac = [2]
            self.kernel_dim = 7
            self.blur_kernel = (1/2**(self.kernel_dim))*torch.ones(self.kernel_dim, self.kernel_dim)
            self.blur_kernel = self.blur_kernel.repeat(1, 3, 1, 1).cuda()

        self.debug = debug
        if self.debug: 
            if path.exists('fixed_z.pt'): 
                self.fixed_z = torch.load('fixed_z.pt')
            else: 
                self.fixed_z = torch.randn(self.batch_size, 512, device='cuda')
                torch.save(self.fixed_z, 'fixed_z.pt') 
                #save fixed z 

    def __len__(self):
        return 5000

    def __getitem__(self, index):
        
        z = torch.randn(1, 512, device='cuda')
        
        if self.debug: 
            z = self.fixed_z
            
        original = self.model(z)[0]
        input_img = original
        
        if self.bootstrap: 
            frac = np.random.rand(1)
            adjusted = self.get_lit_scene(z, frac, self.light_layer, self.light_unit)[0]
            input_img = adjusted
            output_img = original
            frac = -frac
        elif self.debug: 
            output_img = original
            frac = 0
        else: 
            frac = np.random.rand(1)*2-1
            if self.frac_one: 
                frac = [1]
            adjusted = self.get_lit_scene(z, frac, self.light_layer, self.light_unit)[0]
            output_img = adjusted
        

        feat = 0
        if self.use_loc: 
            diff = original.unsqueeze(0) - self.get_lit_scene(z, self.loc_frac, self.light_layer, self.light_unit)
            blur = F.conv2d(diff, self.blur_kernel)
            feat = ((blur) > 0.7).float() * 1
            
            
        data = {'label': input_img, 'image': output_img, 'inst': 0, 'feat': feat, 'path': f'bedroom_{self.num}', 'frac': frac}       
        
        self.num += 1
        return data
    
    def get_lit_scene(self, z, frac, layername, unitnum):
        def change_light(output):
            output.style[:, int(unitnum)] = 10 * frac[0]
            return output
        with nethook.Trace(self.model, f'{layername}.sconv.mconv.modulation', edit_output=change_light):
            return self.model(z)
        
class MultichannelStyleGANDataset(Dataset): 
    def __init__(self,
                 num_stylechannels, 
                 dset='bedroom', 
                 debug = False,
                 reverse = False
                ): 
        self.model = load_seq_stylegan('bedroom', mconv='seq', truncation=0.90)
        nethook.set_requires_grad(False, self.model)
        self.layers = ['layer8', 'layer8']
        self.units = [265, 397] #[lamp, window]

        self.color = torch.tensor([1.0, 1.0, 1.0]).float().cuda()[:,None,None]
        self.num = 0
        self.num_stylechannels = num_stylechannels
        self.reverse = reverse
        self.debug = debug
        if self.debug: 
            if path.exists('fixed_z.pt'): 
                self.fixed_z = torch.load('fixed_z.pt')
            else: 
                self.fixed_z = torch.randn(self.batch_size, 512, device='cuda')
                torch.save(self.fixed_z, 'fixed_z.pt') 
                #save fixed z 

    def __len__(self):
        return 5000

    def __getitem__(self, index):
        
        z = torch.randn(1, 512, device='cuda')
        
        if self.debug: 
            z = self.fixed_z
            
        original = self.model(z)[0]
        frac = np.random.rand(self.num_stylechannels)#*2-1
        adjusted = self.get_lit_scene(z, frac, self.layers, self.units)[0]
        
        if self.reverse: 
            frac = -frac
            input_img = adjusted
            output_img = original
            
        data = {'label': original, 'image': adjusted, 'inst': 0, 'feat': 0, 'path': f'bedroom_{self.num}', 'frac': frac}
        self.num += 1
        return data
    
    def get_lit_scene(self, z, fracs, layers, units):
        #TODO: modify for multiple layers
        layername = layers[0]
        def change_light(output):
            for frac, unit in zip(fracs, units): 
                output.style[:, int(unit)] = 10 * frac
            return output
        with nethook.Trace(self.model, f'{layername}.sconv.mconv.modulation', edit_output=change_light):
            return self.model(z)
        
        
class AlternateStyleGANDataset(Dataset): 
    def __init__(self, 
                 dset='bedroom', 
                 reverse = False
                ): 
        self.model = load_seq_stylegan('bedroom', mconv='seq', truncation=0.90)
        nethook.set_requires_grad(False, self.model)
        self.light_layer = 'layer8'
        self.light_unit = 265
        self.reverse = reverse
        self.num=0
    def __len__(self):
        return 5000

    def __getitem__(self, index):
        
        z = torch.randn(1, 512, device='cuda')
            
        original = self.model(z)[0]
        input_img = original
        
        frac = np.random.rand(1)
        adjusted = self.get_lit_scene(z, frac, self.light_layer, self.light_unit)[0]
        output_img = adjusted
        
        if self.reverse: 
            frac = -frac
            input_img = adjusted
            output_img = original
 
        data = {'label': input_img, 'image': output_img, 'inst': 0, 'feat': 0, 'path': f'bedroom_{self.num}', 'frac': frac}       
        self.num+=1
        return data
    
    def get_lit_scene(self, z, frac, layername, unitnum):
        def change_light(output):
            output.style[:, int(unitnum)] = 10 * frac[0]
            return output
        with nethook.Trace(self.model, f'{layername}.sconv.mconv.modulation', edit_output=change_light):
            return self.model(z)
        
class MaskedStyleGANDataset(Dataset): 
    def __init__(self, 
                 dset='bedroom', 
                 reverse = False
                ): 
        self.model = load_seq_stylegan('bedroom', mconv='seq', truncation=0.90)
        nethook.set_requires_grad(False, self.model)
        self.light_layer = 'layer8'
        self.light_unit = 265
        self.reverse = reverse
        self.num=0
        self.segmodel, _ = load_segmenter()
        
        dim = 11
        blur_kernel = (1/2**(dim))*torch.ones(dim, dim)
        self.blur_kernel = blur_kernel[None, None]
    def __len__(self):
        return 5000

    def __getitem__(self, index):
        z = torch.randn(1, 512, device='cuda')
            
        original = self.model(z)[0]
        input_img = original
        
        mask = self.get_mask(original)
        frac = np.random.rand(1)
        adjusted = self.get_lit_scene(z, mask, frac)[0]
        output_img = adjusted
        
        if self.reverse: 
            frac = -frac
            input_img = adjusted
            output_img = original
         
        #print('output size', output_img.shape)
        #print('input size', input_img.shape)
        data = {'label': input_img, 'image': output_img, 'inst': 0, 'feat': mask[None], 'path': f'bedroom_{self.num}', 'frac': frac}       
        self.num+=1
        return data
    
    def get_mask(self, img): 
        with torch.no_grad():
            _, lamps = self.segmodel.predict_single_class(img[None], 21)
            seg = 1*lamps
        mask = self.get_max_lamp(seg).float().cpu()    
        #mask = (seg[0][0] == 21).float().cpu()
        blurred = F.conv2d(mask[None][None], self.blur_kernel, padding=5).squeeze()
        blurred[blurred!=0] = 1
        return blurred
        
       
    def get_max_lamp(self, seg): #gets mask of largest lamp
        seg = seg.squeeze().int()#[:, 0]
        #print('seg shape', seg.shape)
#         seg[seg!=21] = 0 #lamp index in segmentation = 21
#         seg[seg==21] = 1
        
        binary_lamps = np.uint8(seg.cpu())
        lamp_centroids = []
        max_lamps = [] 

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_lamps, connectivity=8)
        sizes = stats[:, -1]

        if len(np.unique(output))<=1: #no lamps detected
            return torch.Tensor(np.zeros([seg.shape[1], seg.shape[1]]))
        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        max_lamp = np.zeros(output.shape)
        max_lamp[output == max_label] = 1
        return torch.from_numpy(max_lamp)

        
    def get_lit_scene(self, z, mask, frac):
        masked_model = make_masked_stylegan(self.model, z, mask, frac)
        with torch.no_grad(): 
            relit = masked_model(z)#[0]
        return relit



