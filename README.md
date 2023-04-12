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

## Train on example data

Train on the provided example data (note: for some scenes a vanilla nerf must be trained first):\
`cd ../..`\
`python train.py --config configs/castle_nerf.txt`\
`python train.py --config configs/castle.txt`

## Inference on example data

Create some animations from novel view points and decompositions of the Zoo scene:\
`python train.py --config configs/zoo.txt --render_only --render_modes play_vid`

Stream a movie onto the Castle scene:\
`python train.py --config configs/castle.txt --render_only --render_modes play_vid`

Perform XRAY on Teapot & Neko scene:\
`python train.py --config configs/castle.txt --render_only --render_modes play_vid`

Text->Projection on Planck scene:\
`python train.py --config configs/planck.txt --render_only --render_modes multi_t2t --cdc_conda /path/to/cdc/conda/env --cdc_src /path/to/cdc/src`

Note1: for this to work you must have [CDC](https://github.com/cross-domain-compositing/cross-domain-compositing) installed in a seperate folder (where /path/to/cdc/conda/env can be the current one if you installed cdc in the same environment as nepmap).

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
If you deviate from the current scheme of projecting 3 patterns per view where one is white flood-filled, one is black flood-filled, and the last is any random pattern, some adjustments need to be made to the code (data/data_loader.py).

Note about background removal:\
Acquired images must have RGBA information (alpha channel seperating background from foreground). If it doesn't, consider segmenting the background using [rembg](https://github.com/danielgatis/rembg) or other tools.

## Contributions / Requests

Feel free to ask for any additional information or to contribute to the project.


