# MPAC

## 1. Setup
```powershell
conda create -n mpac python=3.8 -y
conda activate mpac
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirement.txt
```

## 2. Models

Download models from https://huggingface.co/flashingtt/MPAC

Git LFS setup
```powershell
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

Download models
```powershell
mkdir saved_models
cd saved_models
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/flashingtt/MPAC
cd MPAC
git lfs pull fiq_img2txt.pt tuned_clip_best.pt image_encoder.pt
```

## 3. Datasets

## 4. Run
### Train
```powershell
# train stage one
sh train_final_s1.sh
# train stage two
sh train_final_s2.sh
```

### Validation
```powershell
sh val_final.sh
```
