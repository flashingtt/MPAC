# MPAC

### Multi-modal Prompt Interaction and Adaptive Feature Composition for Composed Image Retrieval

## Getting Started

### Installation

1. Clone the repo

```sh
git clone https://github.com/flashingtt/MPAC.git
```

2. Install environment

```sh
conda create -n mpac python=3.8 -y
conda activate mpac
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirement.txt
```

## Preparing

### Prepared Models

Download models from https://huggingface.co/flashingtt/MPAC

1. Git LFS setup

```sh
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

2. Download models

```sh
mkdir saved_models
cd saved_models
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/flashingtt/MPAC
cd MPAC
git lfs pull --include="fiq_img2txt.pt tuned_clip_best.pt image_encoder.pt"
```

### Prepare Datasets

1. Get FashionIQ from 

2. Get CIRR from

* To properly work with the codebase FashionIQ and CIRR datasets should have the following structure:

```
dataset_path
└───  fashionIQ_dataset
      └─── captions
            | cap.dress.test.json
            | cap.dress.train.json
            | cap.dress.val.json
            | ...
            
      └───  images
            | B00006M009.jpg
            | B00006M00B.jpg
            | B00006M6IH.jpg
            | ...
            
      └─── image_splits
            | split.dress.test.json
            | split.dress.train.json
            | split.dress.val.json
            | ...

└───  cirr_dataset  
       └─── train
            └─── 0
                | train-10108-0-img0.png
                | train-10108-0-img1.png
                | train-10108-1-img0.png
                | ...
                
            └─── 1
                | train-10056-0-img0.png
                | train-10056-0-img1.png
                | train-10056-1-img0.png
                | ...
                
            ...
            
       └─── dev
            | dev-0-0-img0.png
            | dev-0-0-img1.png
            | dev-0-1-img0.png
            | ...
       
       └─── test1
            | test1-0-0-img0.png
            | test1-0-0-img1.png
            | test1-0-1-img0.png 
            | ...
       
       └─── cirr
            └─── captions
                | cap.rc2.test1.json
                | cap.rc2.train.json
                | cap.rc2.val.json
                
            └─── image_splits
                | split.rc2.test1.json
                | split.rc2.train.json
                | split.rc2.val.json
```

3. Change dataset path ```data_path``` in ```dataset/data_utils.py```

## Running

### Training

1. Training stage one

```sh
sh train_final_s1.sh
```

2. Training stage two

```sh
sh train_final_s2.sh
```

### Validation

```sh
sh val_final.sh
```
