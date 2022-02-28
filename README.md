# DiffeomorphicDA
Implementation of the Data Augmentation (DA) method proposed in the work: Diffeomorphic Transforms for Data Augmentation of Highly Variable Shape and Texture Objects

## Morphing

The source code of the Morphing-based DA method can be found in src/generate_MorphingDA_images.py

## Stationary Velocity Field (SVF)

We provide the configuration of the Stationary Velocity Field registration method in svf. To use it, please have a look at https://github.com/uncbiag/mermaid

## Diffeomorphic Log Demons Image Registration

Use the code at https://www.mathworks.com/matlabcentral/fileexchange/39194-diffeomorphic-log-demons-image-registration

## 2D and 3D spline-based image registration

Use the code at https://github.com/stellaccl/cdmffd-image-registration

## Matching with CNNs

Use the code at https://github.com/ignacio-rocco/cnngeometric_matconvnet and train the model with your dataset

# Training and Test

We also provide train.py and test.py files to train a set of classifiers and test them. It requires PyTorch. 
