

# Local Relighting of Real Scenes
**Audrey Cui, Ali Jahanian, Agata Lapedriza, Antonio Torralba, Shahin Mahdizadehaghdam, Rohit Kumar, David Bau** <br>
arxiv link coming soon!
<br><br>
![teaser image](https://github.com/audreycui/relight/blob/main/docs/teaser.jpg)

Can we use deep generative models to edit real scenes “in the wild”? We introduce the task of local relighting, which changes a photograph of a scene by switching on and off the light sources that are visible within the image. This new task differs from the traditional image relighting problem, as it introduces the challenge of detecting light sources and inferring the pattern of light that emanates from them. 
<br><br>
We propose using a pretrained generator model to generatean unbounded dataset of paired images to train an image-to-image translation model to relight visible light sources in real scenes. In the examples above, the original photo is highlighted with a red border. We show that our unsupervised method is able to turn on light sources
present in the training domain (i.e. lamps) in (a). Additionally, our method can detect and adjust prelit light sources that are way outside
its training domain, such as fire fans (b), traffic lights (c), and road signs (d).

## Setting up the relight_env environment
To run our scripts or notebooks, first create a conda environment by running the following: 
```
conda env create --file=relight_env.yml
```

## Relighting Scripts
To relight a folder of images using our unsupervised method, run

```
python test.py --name unsupervised --netG modulated --no_instance --input_nc 3 --label_nc 0 --dataroot [PATH/TO/DATA] --which_epoch 200 
```

## Interactive notebooks

- ```unsupervised.ipynb```: This notebook contains an interactive demo for our unsupervised method. Add your own test images to ```test_images``` and change the image path in the notebook to run our unsupervised method on your image. 
- ```user_selective.ipynb```: This notebook contains an interactive demo for our user selective method. Likewise, you may add your own test images.


## Training 
To train our modified version of pix2pixHD, run
```
python train.py --name [NAME] --netG modulated --batchSize 8 --max_dataset_size 2000 --no_instance --generated true --label_nc 0 --niter 200 --alternate_train true

```
- ```--name```: Name of the folder this model is saved to (or loaded from) <br>
- ```--netG```: Type of generator. ```modulated``` is our version for relighting. ```global``` is the default from the original pix2pixHD paper. <br> 
- ```--no_instance```: Include this flag if instance maps (see original pix2pixHD code) are not being used. During training of our user selective method, the mask is treated as an instance map and this flag is not used. In all other experiments, including our unsupervised method, this flag is used.  <br>
- ```--generated```: Include this flag if using a generated dataset. otherwise, --dataroot should be used to specify the path to the real images <br>
- ```--alternate_train```: Reverses training sample with negated modulation during training, which results in improvements in turning off lights. See paper for more details. 
- ```--dataroot```: include path to data if using real data for training/testing. We don't need this for using generated data.
- See base_options.py for more general options, train_options.py for more training specific options


## Acknowledgments
- Our code borrows heavily from [pix2pixHD](https://tcwang0509.github.io/pix2pixHD/) for its pix2pix architecture.
- Our code borrows from [rewriting](https://github.com/davidbau/rewriting) for its utility functions.
- We thank the authors of [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch) and [Stylegan2 ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), [encoder4editing](https://github.com/omertov/encoder4editing), and [LPIPS](https://github.com/richzhang/PerceptualSimilarity).
- We thank Daksha Yadav for her insights, encouragement, and valuable discussions
- We are grateful for the support of DARPA XAI (FA8750-18-C-0004) and Signify Lighting Research.
