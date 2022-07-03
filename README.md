# VideoSaliency
PyTorch implementation of a video saliency detection model.

## Usage
For training DHF1K dataset was used. The proper paths should be passed to training script as path arguments.

For evaluating the models weight files are needed, which are excluded of this repo, due to their size. Weight files for ViNet and TASED-Net models can be downloaded from their official repositories linked below. Final weights file will be added once research is finished.

## References
- For the encoder the [S3D](https://github.com/kylemin/S3D) network is used. Also the weights file for training the
encoder network comes from the same source.
- [ViNet](https://github.com/samyak0210/ViNet) main inspiration for the proposed model. The model structure is nearly the same as the 8 frames version of ViNet, but some 3D CNNs are replaced with SepConv blocks proposed by S3D authors.
- [TASED-Net](https://github.com/MichiganCOG/TASED-Net) was one of the main inspirations for the model. It is also one of the models used for evaluation and comparison.
- [DHF1K dataset](https://github.com/wenguanwang/DHF1K) was the main dataset used for training and comparisons between
mentioned models and the proposed one.
- [ACLNet PyTorch implementation](https://github.com/Nablax/ACLnet-Pytorch) is a more modern version of Wang's et. al.
ACLNet which was also referenced during the development.
