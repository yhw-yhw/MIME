## MIME: Human-Aware 3D Scene Generation

<img src="data/MIME.gif" alt="drawing" width="720"/>

This repository contains the code and dataset that accompanies our paper [MIME: Human-Aware 3D Scene Generation
](https://mime.is.tue.mpg.de/).

You can find detailed usage instructions for training your own models, using
our pretrained models as well as performing the interactive tasks described in
the paper below.

If you found this work influential or helpful for your research, please consider citing

```
@inproceedings{yi2022mime,
title = {{MIME}: Human-Aware {3D} Scene Generation},
author = {Yi, Hongwei and Huang, Chun-Hao P. and Tripathi, Shashank and Hering, Lea and 
Thies, Justus and Black, Michael J.},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
month={June}, 
year={2023} 
}
```

## Preliminaries
- See [docs/installation.md](docs/installation.md) to install all the required packages and pretrained models.
- See [docs/dataset.md](docs/dataset.md) to download the 3D-FRONT-Human datasets and learn how to add free-space and contact humans inside a 3D room.

### Download the CHECKPOINT and DATA for running the code.

Please download several files from [the download webpage](https://mime.is.tue.mpg.de/login.php).

* Downlad the [MIME_CKPT](https://download.is.tue.mpg.de/download.php?domain=mime&resume=1&sfile=MIME_CKPT.tar.gz) and put it into `data/CKPT`
* Download other needed [code_data](https://download.is.tue.mpg.de/download.php?domain=mime&resume=1&sfile=code_data.tar.gz) and put them into `data`.
* Download 3D-FRONT-HUMAN preprocess data [preprocess_3DFRONTHUMAN_input.tar.gz](https://download.is.tue.mpg.de/download.php?domain=mime&resume=1&sfile=preprocess_3DFRONTHUMAN_input.tar.gz) and unzip into `data/preprocess_3DFRONTHUMAN_input`
* Download our preprocessed [3DFRONTHUMAN_relative_path_pkl.zip](https://download.is.tue.mpg.de/download.php?domain=mime&resume=1&sfile=3DFRONTHUMAN_relative_path_pkl.zip) for 3D-FRONT-Human and unzip into `data/3DFRONTHUMAN_relative_path_pkl`
* Download our preprocess body input [input_finement](https://download.is.tue.mpg.de/download.php?domain=mime&resume=1&sfile=input_refinement.tar.gz) for running scene refinement, unzip it into `data/input_refinement`.
* Download several visualized [samples](https://download.is.tue.mpg.de/download.php?domain=mime&resume=1&sfile=ThreeD-FRONT-HUMAN-samples.tar.gz) in 3D-FRONT-HUMAN for visualization. 

### Change the environment variables.

You need to modify these variables in `env.sh`.

```
# ! need to modify.
export PYTHONPATH=${change_to_python_path}:$PYTHONPATH
export CODE_ROOT_DIR=${change_to_code_path}
export DATA_ROOT_DIR=${change_to_original_3DFRONT_path}
```

## Scene Generation

- See [docs/inference.md](docs/inference.md) to generate scenes from input humans.


## Scene Refinement

- See [docs/refinement.md](docs/refinement.md) to refine the scene layout with gemetrical human-scene interaction details.

## Visualize the Dataset and Results

- See [docs/visualization.md](docs/visualization.md) to visualize the 3D-FRONT-HUMAN dataset and the distribution of different objects locations in our MIME generated results.  

## Training

- See [docs/training.md](docs/train.md) to train your own models.

## Evaluation

- See [docs/evaluation.md](docs/evaluation.md) to benchmark your pretrained models.

## Acknowledgments

```
We thank Despoina Paschalidou, Wamiq Para for useful feedback about the reimplementation of ATISS, 
and Yuliang Xiu, Weiyang Liu, Yandong Wen, Yao Feng for the insightful discussions, 
and Benjamin Pellkofer for IT support. 
This work was supported by the German Federal Ministry of Education and Research (BMBF): Tübingen AI Center, FKZ: 01IS18039B.  
```

We build our MIME architecture based on [ATISS](https://github.com/nv-tlabs/ATISS).

## Disclosure

```
MJB has received research gift funds from Adobe, Intel, Nvidia, Meta/Facebook, and Amazon. 
MJB has financial interests in Amazon, Datagen Technologies, and Meshcapade GmbH. 
JT has received research gift funds from Microsoft Research.
```
