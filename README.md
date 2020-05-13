# Deep Learning Registration for Cardiac Motion Tracking

## Introduction
Deep learning network-based registration method applied on cardiac motion tracking from cardiac MR images (cMRI).
If you use this code or part of this code, please consider citing the following papers:
> Qiu, H., Qin, C., Le Folgoc, L., Hou, B., Schlemper, J., Rueckert, D.:   
> **Deep Learning for Cardiac Motion Estimation: Supervised vs. Unsupervised Training**  
> [STACOM Workshop, MICCAI 2019.](https://doi.org/10.1007/978-3-030-39074-7_20)  
> (STACOM19 version of the code can be found in branch [`stacom19`](https://github.com/qiuhuaqi/cardiac-motion/tree/stacom19))

> Qin, C., Bai, W., Schlemper, J., Petersen, S.E., Piechnik, S.K., Neubauer, S., Rueckert, D.:  
> **Joint learning of motion estimation and segmentation for cardiac MR image sequences**  
> [MICCAI 2018](https://doi.org/10.1007/978-3-030-00934-2_53)

## Instructions
### Dependencies
Code developed and tested on Ubuntu 16.04 & 18.04 operating systems, using Python 3.6 and Pytorch 1.0.

To install the Python dependencies, run the following in the root directory of the repo after cloning the repo:
```
pip3 install -r requirements.txt
```
CUDA and cuDNN are required (tested with CUDA `9.0.176` and cuDNN `7.1.4`). 
The code should work with any CUDA and cuDNN versions supported by Pytorch 1.0. Please refer to Pytorch and NVIDIA websites.


### Running
The code works on a model-directory-basis. Training, testing and inference of a model are all based on the model directory of this model. 
Logs, trained models, testing and inference results are all saved in the model directory. 

Training:
```
python cardiac_motion/train.py --gpu [gpu_num] --model_dir [path_to_model_dir]
```

Testing (on the end-diastolic and end-systolic frames): 
```
python cardiac_motion/eval.py --gpu [gpu_num] --model_dir [path_to_model_dir] --restore_file [file_name_of_saved_model]
```

Inference (on all frames of the sequences):
```
python cardiac_motion/inference.py --gpu [gpu_num] --model_dir [path_to_model_dir] --data_dir [path_to_data_dir]
```

Most setting parameters related to data or model are specified in the `params.json` file, which should be supplied in the model directory. 
This file is parsed into attributes of the object `params` in the code to pass the parameters. An example of this file is provided in the repo root directory.

## Trained models
Models trained on cardiac MR image data from the [UK Biobank Imaging Study](https://imaging.ukbiobank.ac.uk/) is available. 
Please feel free to email us to enquire if you are interested.

## Contact us
If you have any question regarding the paper or the code, feel free to open an issue in this repo or email us at:
huaqi.qiu15@imperial.ac.uk

