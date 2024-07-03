# HNC: Leveraging Hard Negative Captions towards Models with Fine-Grained Visual-Linguistic Comprehension Capabilities

This repository contains the source code of the publication: HNC: Leveraging Hard Negative Captions towards Models with Fine-Grained Visual-Linguistic Comprehension Capabilities at CoNLL 2023 by Esra Dönmez*, Pascal Tilli*, Hsiu-Yu Yang*, Ngoc Thang Vu and Carina Silberer.

https://aclanthology.org/2023.conll-1.24.pdf

## Install Environment
Create a virtual python environment with e.g. anaconda:

```bash
conda create --name hnc python=3.10
```
Activate the just created environment with:
```bash
conda activate hnc
```
Install the requirements via pip:
```bash
pip install -r requirements.txt
```

## Download HNC
Download the automatically generated train and validation set as well as the human annotated test set from DaRUS: https://doi.org/10.18419/darus-4341
or HuggingFace: https://huggingface.co/datasets/patilli/HNC 

## Download GQA
To rerun the dataset creation based on scene graphs of GQA, download the dataset from https://cs.stanford.edu/people/dorarad/gqa/about.html .

## Update Configs
Input the paths for the train and valid split of the GQA scene graphs into the `config_default.json` at following keys: `gqa_sg_train` and `gqa_sg_valid`.

## Run the Script
To run the dataset creation, just execute the main.py via:
```bash
python main.py
```
Optionally, you can pass a `--config` argument followed by the path to your config.
As default, the script uses the `config_default.json`.

## Citation
```bibtex
@inproceedings{hnc,
    title = "{HNC}: Leveraging Hard Negative Captions towards Models with Fine-Grained Visual-Linguistic Comprehension Capabilities",
    author = {D{\"o}nmez, Esra  and
      Tilli, Pascal  and
      Yang, Hsiu-Yu  and
      Vu, Ngoc Thang  and
      Silberer, Carina},
    booktitle = "Proceedings of the 27th Conference on Computational Natural Language Learning (CoNLL)",
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.conll-1.24",
    doi = "10.18653/v1/2023.conll-1.24",
    pages = "364--388",
}
```
