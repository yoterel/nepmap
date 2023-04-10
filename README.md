# Neural Projection Mapping Using Reflectance Fields

This is the official implmentation of "Neural Projection Mapping Using Reflectance Fields".

## Installation

Clone the repository :\
`git clone https://github.com/yoterel/nepmap.git`

Create a conda environment :\
`conda env create -n nepmap --file environment.yml`

Activate the environment :\
`conda activate nepmap`

Install nerfacc from the ext folder (note: custom changes have been made to occ grid class so this version must be used):\
`cd ext/nerfacc`\
`pip install .`

Train on the provided example data (note: for some scenes a vanilla nerf must be trained first):\
`cd ../..`\
`python train.py --config configs/castle_nerf.txt`\
`python train.py --config configs/castle.txt`

Run inference on the provided example data :\
`python train.py --config configs/castle_inference.txt`

Train on your own data :\
You must supply a dataset folder in similar structure to the example data.

Note about background removal:\
The raw acquired images must have RGBA information. If it doesn't, consider segmenting the background using [rembg](https://github.com/danielgatis/rembg) 


