[Project Page](https://yoterel.github.io/nepmap-project-page/) | [Paper](#) | [Supplementary](https://yoterel.github.io/nepmap-project-page/static/pdfs/nepmap-supp.pdf) | [Dataset](https://osf.io/2xcn4/download) | [Pretrained Models](https://osf.io/nzsk8/download)

# Neural Projection Mapping Using Reflectance Fields

This is the official implmentation of "Neural Projection Mapping Using Reflectance Fields".

## Installation

Clone the repository :\
`git clone https://github.com/yoterel/nepmap.git`

Create a conda environment :\
`conda env create -n nepmap --file environment.yml`

Activate the environment :\
`conda activate nepmap`

Install nerfacc from the ext folder (note: custom changes have been made to occ grid class so this version of nerfacc must be used):\
`cd ext/nerfacc-0.3.5`\
`pip install .`

Optionally download datasets and pretrained models (place them at nepmap/datasets, nepmap/logs):\
`mkdir datasets`\
`cd datasets`\
`wget https://osf.io/2xcn4/download`

`mkdir logs`\
`cd logs`\
`wget https://osf.io/nzsk8/download`

Tested on Ubuntu 18.04.6, with Nvidia RTX A6000.
For GPUs with less memory, you might need to reduce one of the following hyper parameters for training or inference:
- grid_resolution  # controls occupancy grid resolution for nerfacc
- render_n_samples  # controls number of initial samples per ray (actual samples are determined by the occupancy)
- target_sample_batch_size  # controls the target number of rays per batch (batch size is not fixed)
- test_chunk_size  # chunk size during inference
- _update() function under grid.py  # there is a hard coded thereshold of 800k which splits occupancy queries into chunks. reduce it to avoid OOM.

## Datasets & Pretrained models

Download and place the dataset under nepmap/datasets
Download and place the pretrained models under nepmap/logs

All synthetic scenes were created using the [sandbox.blend](https://github.com/yoterel/nepmap/blob/master/sandbox.blend) file.

## Train on example data

Train on the provided example data (note: for some scenes a vanilla nerf must be trained first):\
`cd ../..`\
`python train.py --config configs/castle_nerf.txt`\
`python train.py --config configs/castle.txt`

## Inference on example data

Decompose the Zoo scene on the training set:\
`python train.py --config configs/zoo.txt --render_only --render_modes train_set_movie`

Stream a movie onto the Castle scene:\
`python train.py --config configs/castle.txt --render_only --render_modes play_vid`

Perform XRAY on Teapot & Neko scene:\
`python train.py --config configs/teapot.txt --render_only --render_modes dual_photo`

Text->Projection on Planck scene:\
`python train.py --config configs/planck.txt --render_only --render_modes multi_t2t --cdc_conda /path/to/cdc/conda/env --cdc_src /path/to/cdc/src`

Text->Projection on Bunny scene:\
`python train.py --config configs/bunny.txt --render_only --render_modes multi_t2t --cdc_conda /path/to/cdc/conda/env --cdc_src /path/to/cdc/src`

Note1: for the text->projection to work you must have [CDC](https://github.com/cross-domain-compositing/cross-domain-compositing) installed in a seperate folder (where /path/to/cdc/conda/env can be the current one if you installed cdc in the same environment as nepmap).

Note2: The current code produces multiple results for different content aware thresholds ("T_in"). The outputs will be produced in the folder diffuse_i/output where i is the current view being processed. You will be prompted to select an output image, name it "diffuse_final.png" and place it in a diffuse_i folder.
This may be fully automated if T_in is a single value.

## Train on your own data

You must supply a dataset folder in similar structure to the example data (for synthetic scenes, see Castle for example).
For real scenes, this code assumes COLMAP is installed and available in the current context, and expects the following folder structure:

    dataset_folder\
        0000.png
        0001.png
        ...
        projector\
            raw_patterns\
                0000.png
                0001.png
                ...

Where dataset_folder contains the RGBA images of the scene, and projector/raw_patterns contains the raw patterns that were projected.
If you deviate from the current scheme of projecting 3 patterns per view where one is white flood-filled, one is black flood-filled, and the last is any random pattern, some adjustments need to be made to the code ([data_loader.py](https://github.com/yoterel/nepmap/blob/master/data/data_loader.py)).

Note about background removal:\
Acquired images must have RGBA information (alpha channel seperating background from foreground). If it doesn't, consider segmenting the background using [rembg](https://github.com/danielgatis/rembg) or other tools.
Optimization should work without an alpha channel, but this greatly reduces the quality of the results.

## Troubleshooting / Requests

Feel free to ask for any additional information by openning a github issue.


