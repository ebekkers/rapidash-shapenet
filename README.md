# ✨ 🐴 🔥 PONITA with minimal library dependencies

## Conda environment
In order to run the code in this repository install the following conda environment
```
conda create --yes --name ponita-shapenet python=3.12 numpy
conda activate ponita-shapenet
pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
pip install torch_geometric==2.5.3
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install tqdm
pip install rdkit
pip install pandas
pip install pytorch_lightning
pip install wandb
pip install chamferdist
```
