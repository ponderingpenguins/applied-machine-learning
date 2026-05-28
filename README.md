# Based on Deep Learning-Based Gait Recognition Using Smartphones in the Wild by Zou Q, Wang Y, Zhao Y, Wang Q and Li Q

https://github.com/qinnzou/Gait-Recognition-Using-Smartphones

## Setup

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1KOm-zROeOZH3e2tqYUpHAvIaBZSJGFm_), and place it in the same directory as this README.

Unzip the dataset

```bash
unzip Gait-Datasets-TIFS20.zip # should be in the same directory as this README
```

You also need to unzip Dataset #1, which is in the `Gait-Datasets-TIFS20` directory.

```bash
cd Gait-Datasets-TIFS20
unzip Dataset\ #1.zip
```

Then, create a virtual environment and install the dependencies:

```bash
uv sync
```

## Training and finetuning

Train a baseline model with:

```bash
python -m gait_classification.train model_type=lstm
python -m gait_classification.train model_type=transformer
```

Run the iterative hyperparameter search with SEM-based comparisons using:

```bash
python -m gait_classification.finetune model_type=lstm
python -m gait_classification.finetune model_type=transformer
```

The finetuning run saves a compact summary and per-stage plots in `checkpoints/finetuning/`.
