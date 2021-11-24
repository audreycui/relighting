import torch
import copy, contextlib
from rewrite_utils import nethook, pbar, zdataset, renormalize, smoothing
from rewrite_utils.stylegan2.models import DataBag
from torchvision.transforms import Resize

class ApplyMaskedStyle(torch.nn.Module):
    def __init__(self, mask, frac):
        super().__init__()
        self.register_buffer('mask', mask)
        #self.register_buffer('fixed_style', fixed_style)
        self.unit = 265
        self.frac = frac
        
    def forward(self, d):
        modulation_light = d.style.detach().clone() 
        modulation_light[:, self.unit] = self.frac*20
        
        modulation_no_light = d.style.detach().clone()        
        #modulation_no_light[:, self.unit] = 0
        
        modulation = (modulation_light[:,:,None,None] * self.mask) + (
            modulation_no_light[:,:,None,None] * (1 - self.mask))
        return DataBag(d, fmap=modulation * d.fmap,
             style=d.style)
    
def make_masked_stylegan(gan_model, initial_z, mask, frac):
    '''
    Given a stylegan and a mask (encoded as a PNG) and an initial z,
    creates a modified stylegan which applies z only to a masked
    region.
    '''
    layer = 'layer8.sconv.mconv.adain'
    shape = [32, 32]
    with torch.no_grad():
        masked_model = copy.deepcopy(gan_model)
        #masked_model = gan_model
        device = next(masked_model.parameters()).device

        parent = nethook.get_module(masked_model, layer[:-len('.adain')])
        #shape = style_shapes[layer][-2:]
        downsize = Resize(shape)       

        mask = mask[None, None, :, :]
        mask = downsize(mask)
        
        if shape[0] > 16:
            sigma = float(shape[0]) / 16.0
            kernel_size = (int(sigma) * 2 - 1)
            blur = smoothing.GaussianSmoothing(1, kernel_size, sigma=sigma)
            mask = blur(mask)
        mask = mask[0, 0].to(device)

        parent.adain = ApplyMaskedStyle(mask, frac[0])
    return masked_model