# MPAC

## 1. Setup

```powershell
conda create -n mpac python=3.8 -y
conda activate mpac
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirement.txt
```