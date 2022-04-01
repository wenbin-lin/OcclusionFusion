# OcclusionFusion (CVPR'2022)

### [Project Page](https://wenbin-lin.github.io/OcclusionFusion/) | [Paper](https://arxiv.org/abs/2203.07977) | [Video](https://youtu.be/UC4k43dlp2w)

## Overview

This repository contains the code for the CVPR 2022 paper [OcclusionFusion](https://arxiv.org/abs/2203.07977), where we introduce a novel method to calculate occlusion-aware 3D motion to guide dynamic 3D reconstruction.

In our technique, the motion of visible regions is first estimated and combined with temporal information to infer the motion of the occluded regions through an LSTM-involved graph neural network.  

Currently, we provide a pretrained model and a demo. Code for data pre-processing, network training and evaluation will be available soon.

## Setup

We use `python 3.8.10`, `pytorch-1.8.0` and `pytorch-geometric-1.7.2`. 

    conda create -n occlusionfu python==3.8.10
    conda activate occlusionfu
    pip install -r requirements.txt
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
    pip install torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
    pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
    pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
    pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
    pip install torch-geometric==1.7.2

## Running the demo

Run the demo with the pretrained model and prepared inputs:

    python demo.py

Visualize the input and output:

    python visualize.py

The defualt setting of `visualize.py` will render the network's input and output to a video as follow. You can also change the setting to view the network's input and output with [Open3D](http://www.open3d.org/) viewer.

<p align="center">
  <img width="50%" src="media/teaser.gif"/>
</p>

## Citation

If you find our work useful in your research, please consider citing:

    @inproceedings{lin2022occlusionfusion,
        title={OcclusionFusion: Occlusion-aware Motion Estimation for Real-time Dynamic 3D Reconstruction}, 
        author={Wenbin Lin, Chengwei Zheng, Jun-Hai Yong, Feng Xu}, 
        journal={Conference on Computer Vision and Pattern Recognition (CVPR)}, 
        year={2022}
    } 
