

# Local Relighting of Real Scenes

## Setting up the relight_env environment
```
conda env create --file=religth_env.yml
```
## Pretrained model
- Our pretrained model can be downloaded from Google Drive [here](https://drive.google.com/drive/folders/1jK52oEfoYcUI_CMw6_wt57Dii5unE502?usp=sharing).
- Run ```make_model_dir.sh``` and place the model files from ```unsupervised``` into ```checkpoints/unsupervised``` and the files from ```selective``` into ```checkpoints/selective```.
## Interactive notebooks
- ```unsupervised.ipynb```: This notebook contains an interactive demo for our unsupervised method. Add your own test images to ```test_images``` and change the image path in the notebook to run our unsupervised method on your image. 
- ```user_selective.ipynb```: This notebook contains an interactive demo for our user selective method. Likewise, you may add your own test images.
## Running training/testing scripts
```
python train.py --name [NAME] --netG modulated --batchSize 8 --max_dataset_size 2000 --no_instance --generated true --label_nc 0 --niter 200 
```
```
python test.py --name [NAME]  --netG modulated --no_instance --input_nc 3 --label_nc 0 --dataroot datasets/lsun_bedrooms/ --which_epoch 200 
```

- ```--name```: name of the folder this model is saved to (or loaded from) <br>
- ```--netG```: type of generator. ```modulated``` is our version for relighting. ```global``` is the default from the original pix2pixHD paper. <br> 
- ```--no_instance```: include this flag if instance maps (see original pix2pixHD code) are not being used. During training of our user selective method, the mask is treated as an instance map and this flag is not used. In all other experiments, including our unsupervised method, this flag is used.  <br>
- ```--generated```: include this flag if using a generated dataset. otherwise, --dataroot should be used to specify the path to the real images <br>
- ```--n_stylechannels```: number of stylechannels that will be modulated. The layers/units of the stylechannels should be specified in custom_dataset_loader.py. <br>
- ```--dataroot```: include path to data if using real data for training/testing. We don't need this for using generated data.
- There are more options in base_options.py for general options, train_options.py for training specific options, test_options for testing specific options 
<br><br>

 
## How the code works: 
### Creating a dataset
- During training/testing, ```CreateDataLoader``` in ```data/dataloader.py``` is called. 
- If ```--generated``` is false and a ```--dataroot``` is provided, an ```AlignedDataset``` (from original pix2pixHD code) is created for the folder of images. 
- If ```--generated``` is true, a StyleGAN2 pretrained on LSUN bedrooms is loaded and ```StyleGANDataLoader``` in ```data/custom_dataset_data_loader.py``` calls appropriate dataset (decided by options flags). 


### pix2pix model
- forward and inference functions are in model/pix2pixHD.py. 
- networks defined in models/networks.py — this is where our ResNet modulation (ModulatedGenerator) is defined.



## Acknowledgments
- This code borrows heavily from [pix2pixHD](https://tcwang0509.github.io/pix2pixHD/) for its pix2pix architecture.
- This code borrows from [rewriting](https://github.com/davidbau/rewriting) for its utility functions.
- We thank the authors of [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch) and [Stylegan2 ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), [encoder4editing](https://github.com/omertov/encoder4editing), and [LPIPS](https://github.com/richzhang/PerceptualSimilarity).
