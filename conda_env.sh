#!/bin/bash --login

set -e

# Execute this on Lambda nodes before you start creating the conda env!
# source /etc/profile.d/lmod.sh
# module avail
# module load cuda/11.8
# which nvcc

# Manually run these commands before running this sciprt
# conda create -n vphylo python=3.10 pip --yes
# conda activate vphylo

# PyTorch w/ CUDA 11.8 (good fit for Tesla V100)
# https://pytorch.org/get-started/previous-versions/
# torch==2.4.0 is currently the latest stable release that ships pre-built CUDA 11.8 wheels (which run cleanly on V100).
# Going higher often forces CUDA 12.x, and V100 is from the CUDA 11 generation → that combination can introduce:
# - driver mismatches
# - worse performance on fp16 on sm_70
# - random “illegal memory access” under mixed precision
# - model checkpoint compatibility oddities
# - more people have tested T5-family LoRA on Torch 2.4 than on 2.5+ with V100s
pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

pip install "transformers>=4.44" peft accelerate datasets einops sentencepiece tokenizers
pip install biopython scikit-learn pandas numpy tqdm
pip install scikit-bio
# (optional, for tree plotting later)
pip install ete3 matplotlib

# Other
# conda install -c conda-forge ipdb=0.13.9 --yes
# conda install -c conda-forge python-lsp-server=1.2.4 --yes
##pip install ipywidgets

# Check installs
# python -c "from datasets import load_dataset; print(load_dataset('squad', split='train')[0])"
# python -c "import torch; print(torch.__version__)"
# python -c "import torch; print(torch.version.cuda)"
# python -c "import datasets; print(datasets.__version__)"
# python -c "import torch_geometric; print(torch_geometric.__version__)"
# python -c "import networkx; print(networkx.__version__)"
# python -c "import matplotlib; print(matplotlib.__version__)"
# python -c "import h5py; print(h5py.version.info)"
# python -c "import pubchempy; print(pubchempy.__version__)"
# python -c "import rdkit; print(rdkit.__version__)"
