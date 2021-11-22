

# modulated pix2pix

## running training/testing scripts
```
python train.py --name [NAME] --netG modulated --batchSize 8 --max_dataset_size 2000 --no_instance --generated true --label_nc 0 --niter 200 --n_stylechannels [NUMCHANNELS] --use_location_map true
```
```
python test.py --name [NAME]  --netG modulated --no_instance --input_nc 3 --label_nc 0 --dataroot datasets/lsun_bedrooms/ --which_epoch 40 --n_stylechannels 2
```

- name: name of the folder this model is saved to (or loaded from) <br>
- netG: type of generator. modulated is my version. global is the default from the original paper. <br> 
- no_instance: include this flag if instance maps (gets concatenated with input, pix2pix's way of conditioning) are not being used
- generated: include this flag if using a generated dataset. otherwise, --dataroot should be used to specify the path to the real images <br>
- n_stylechannels: number of stylechannels that will be modulated. as of now, the layers/units of the stylechannels must be specified in ```custom_dataset_loader.py```. <br>
- use_location_map: adds location of lamps as a feature map, which gets concatenated to the rest of the input. currently only supported for single channel modulations <br>
- dataroot: include path to data if using real data for training/testing. don't need this if using generated data.
- more options in base_options.py for general options, train_options.py for training specific options, test_options for testing specific options 
<br><br>

## how the code works: 
### creating a dataset
- calls CreateDataLoader in data/dataloader.py
- for generated datasets, StyleGANDataLoader in data/custom_dataset_data_loader.py calls appropriate dataset (decided by options flags)
- easiest way to experiment is to define a new dataset in custom_data_loader and have StyleGANDataLoader to call that

### pix2pix model
- forward and inference functions in model/pix2pixHD.py. 
- networks defined in models/networks.py — this is where resnet modulation (ModulatedGenerator) is defined


## Acknowledgments
This code borrows heavily from [pix2pixHD](https://tcwang0509.github.io/pix2pixHD/).
