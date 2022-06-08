import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

import random 
import numpy as np 

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website

dataroot = opt.dataroot.replace("/", "_")
generated = 'generated' if opt.generated else f'real_{dataroot}'

web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_%s_%s' % (opt.phase, opt.which_epoch, generated, fraction))
print(web_dir)
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx
    
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst']  = data['inst'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1 

                           
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']]) 
    elif not opt.generated: 
        frac = np.random.rand(opt.n_stylechannels)*2-1
        
        generated = model.inference(data['label'], data['inst'], data['image'], amount=frac)
        visuals = OrderedDict([('input_image', util.tensor2im(data['label'][0])),
                               ('output_image', util.tensor2im(generated.data[0])), 
                               ])
    else:  
        frac = data['frac']
        generated = model.inference(data['label'], data['inst'], data['image'], amount=frac)
        visuals = OrderedDict([('input_image', util.tensor2im(data['label'][0])),
                               ('stylespace_target', util.tensor2im(data['image'][0])), 
                               ('output_image', util.tensor2im(generated.data[0])), 
                               ])

    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path, idx=i, scalar=frac)

webpage.save()
