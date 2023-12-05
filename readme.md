# HNC: Leveraging Hard Negative Captions towards Models with Fine-Grained Visual-Linguistic Comprehension Capabilities

This repository contains the source code of the publication: HNC: Leveraging Hard Negative Captions towards Models with Fine-Grained Visual-Linguistic Comprehension Capabilities at CoNLL 2023 by Esra Dönmez, Pascal Tilli, Hsiu-Yu Yang, Ngoc Thang Vu and Carina Silberer.

## Install Environment
Create a virtual python environment with e.g. anaconda:

```
conda create --name hnc python=3.10
```
Activate the just created environment with:
```
conda activate hnc
```
Install the requirements via pip:
```
pip install -r requirements.txt
```

## Download GQA
To rerun the dataset creation based on scene graphs of GQA, download the dataset from https://cs.stanford.edu/people/dorarad/gqa/about.html .

## Update Configs
Input the paths for the train and valid split of the GQA scene graphs into the `config_default.json` at following keys: `gqa_sg_train` and `gqa_sg_valid`.

## Run the Script
To run the dataset creation, just execute the main.py via:
```
python main.py
```
Optionally, you can pass a `--config` argument followed by the path to your config.
As default, the script uses the `config_default.json`.
