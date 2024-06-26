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

1. Get FashionIQ from https://github.com/XiaoxiaoGuo/fashion-iq

2. Get CIRR from https://github.com/Cuberick-Orion/CIRR

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

1. Training stage one ```sh train_final_s1.sh```

```sh
python clip_fine_tune.py \
   --dataset fashioniq \
   --experiment-name fiq_clip_vitb16_231229_test \ # your experiment name
   --num-epochs 100 \
   --clip-model-name ViT-B/16 \
   --encoder none \
   --learning-rate 2e-6 \
   --batch-size 128 \
   --transform targetpad \
   --target-ratio 1.25 \
   --save-training \
   --save-best \
   --validation-frequency 1 \
   --network clip4cir_maple_final_s1 \
   --final \
   --num-workers 64 \
   --fixed-image-encoder \
   --img2txt-model-path ./saved_models/MPAC/fiq_img2txt.pt \ # to get pseudo token s^*
   --asynchronous \
   --clip-model-path ./saved_models/MPAC/fiq_tuned_clip_best.pt \ # tuned clip model path
   --clip-image-encoder-path ./saved_models/MPAC/fiq_image_encoder.pt \ # to get pseudo token s^*
   --maple-prompt-depth 9 \
   --maple-ctx-init 'a photo of' \
   --maple-n-ctx 3
```

2. Training stage two ```sh train_final_s2.sh```

```sh
CUDA_VISIBLE_DEVICES=5 \
python combiner_train.py \
   --dataset fashioniq \
   --experiment-name fiq_comb_ViT-B16_fullft_231230_test \ # your experiment name
   --projection-dim 4096 \
   --hidden-dim 8192 \
   --num-epochs 300 \
   --clip-model-name ViT-B/16 \
   --combiner-lr 2e-5 \
   --batch-size 4096 \
   --clip-bs 32 \
   --transform targetpad \
   --target-ratio 1.25 \
   --save-training \
   --save-best \
   --validation-frequency 1 \
   --network clip4cir_maple_final_s2 \
   --combiner combiner_v5 \
   --final \
   --num-workers 64 \
   --fixed-image-encoder \
   --img2txt-model-path ./saved_models/MPAC/fiq_img2txt.pt \ # to get pseudo token s^*
   --model-s1-path ./models/clip_finetuned_on_fiq_ViT-B/16_2023-12-29_11:43:23/saved_models/best_model.pt \ # saved model path in stage one
   --asynchronous \
   --optimizer combiner \
   --mu 0.1 \
   --router \
   --maple-prompt-depth 1 \
   --maple-ctx-init 'a photo of' \
   --maple-n-ctx 3
```

### Validation

```sh val_final.sh```

```sh
python validate.py \
   --dataset fashioniq \
   --model-s1-path ./models/combiner_trained_on_fiq_ViT-B/16_2023-12-27_01:35:11/saved_models/model.pt \ # saved model path in stage one
   --projection-dim 4096 \
   --hidden-dim 8192 \
   --clip-model-name ViT-B/16 \
   --target-ratio 1.25 \
   --transform targetpad \
   --network clip4cir_maple_final_s2 \
   --final \
   --num-workers 64 \
   --maple-n-ctx 3 \
   --maple-ctx-init 'a photo of' \
   --maple-prompt-depth 9 \
   --asynchronous \
   --fixed-image-encoder \
   --combiner combiner_v5 \
   --model-s2-path ./models/combiner_trained_on_fiq_ViT-B/16_2023-12-27_01:35:11/saved_models/combiner.pt \ # saved model path in stage two
   --img2txt-model-path ./saved_models/MPAC/fiq_img2txt.pt \
   --optimizer combiner
```
