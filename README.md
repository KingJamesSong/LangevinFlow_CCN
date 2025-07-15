# Langevin Flows for Modeling Neural Latent Dynamics


# Usage

## Dataset Setup
### Downloading from DANDI Archive
First, install the `dandi` CLI (`pip install dandi`) and download each of the NLB datasets into the same folder (e.g. `dandi download DANDI:000128/0.220113.0400` for `mc_maze`). We will refer to this folder as `DATA_HOME`. Edit the `.env` file in the `nlb-lightning` repo to set this path appropriately. Download commands can be found on the DANDI Archive page for each dataset (links on the [NLB Datasets page](https://neurallatents.github.io/datasets)).

### Preprocessing
We define a `main` function in the `scripts/preprocess.py` module that preprocesses and saves data for training, using the functions in `nlb_tools.make_tensors`. This module can be run as a script to generate all 28 splits used for NLB. The preprocessed data will be stored in the `PREP_HOME` folder (edit `.env` accordingly), with a unique folder for each split (`{dataset_name}-{bin_width:02}ms-{phase}`). Within each split folder, the filenames used for each type of data are specified by `TRAIN_INPUT_FILE`, `EVAL_INPUT_FILE`, and `EVAL_TARGET_FILE` in `.env`. The `NLBDataModule` will load data directly from these files.



## Model Training
The training scripts for NLB subsets are `scripts/train_langevin_maze.py`, `scripts/train_langevin_bump.py`, `scripts/train_langevin_rtt.py`, and `scripts/train_langevin_rsg.py`. Training logs and checkpoints will be stored in the `RUNS_HOME` directory that you specify in `.env`. Inside `RUNS_HOME`, each run will be stored at `{run_tag}/{data_tag}`, where `data_tag` is resolved using the f-string in the [Preprocessing section](#preprocessing). After training, the output of each model in a `run_tag` directory is written to a shared `submission-{phase}.h5` file, which can be uploaded directly to EvalAI.


## Citation

```
@article{song2025langevin,
  title     = {Langevin Flows for Modeling Neural Latent Dynamics},
  author    = {Song, Yue and Keller, T. Anderson and Yue, Yisong and Perona, Pietro and Welling, Max},
  booktitle = {Cognitive Computational Neuroscience (CCN)},
  year      = {2025}
}
```
