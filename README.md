# On the Robustness of Vision Transformers

We provide code for attacking a single Vision Transformer (ViT-L-16), a Big Transfer Model (BiT-M-R101x3) or a combination (ViT + BiT) defense presented in the original paper: https://arxiv.org/abs/2104.02610.
All attacks provided here are done on CIFAR-10 using PyTorch.
With the proper parameter selection and models, this same code can also be easily re-tooled for CIFAR-100 and ImageNet. 
Each attack can be run by uncommenting one of the lines in the main. 

We provide attack code for the Self-Attention Gradient Attack (SAGA), the Adaptive attack, and a wrapper for using the RayS attack (original RayS attack code here: https://github.com/uclaml/RayS)

# Step by Step Guide

<ol>
  <li>Install the packages listed in the Software Installation Section (see below).</li>
  <li>Download the "Models" from the Google Drive link listed in the Models Section.</li>
  <li>Move the Models folder into the directory ".\VisionTransformersRobustness\VisionTransformersRobustness"</li>
  <li>Open the VisionTransformersRobustness.sln file in the Python development of your choice. Choose one of the attack lines and uncomment it. Run the main.</li>
</ol>

# Software Installation 

We use the following software packages: 
<ul>
  <li>pytorch==1.7.1</li>
  <li>torchvision==0.8.2</li>
  <li>numpy==1.19.2</li>
  <li>opencv-python==4.5.1.48</li>
</ul>

# Models

We provide the following models:
<ul>
  <li>ViT-L-16</li>
  <li>BiT-M-R101x3</li>
  <li>Google's Pretrained ViT-B-32</li>
</ul>

The models can be downloaded here: https://drive.google.com/drive/folders/1Zy5DeCxU2KoPXx3TETzfiwbuHimMYO_9?usp=sharing
The ViT or BiT-M models are necessary to run any of the attacks. The ViT-B-32 model from Google is only needed for the Adapative attack, as it is used as the starting synthetic model.


# System Requirements 

All our attacks are tested in Windows 10 with 12 GB GPU memory (Titan V GPU). The Adaptive attack has additional hardware requirements. To run this attack you need 128 GB RAM, and at least 200 GB of free hard disk space.  

# Contact

For questions or concerns please contact the author at: kaleel.mahmood@uconn.edu

