
<div align="center">
    <h1  style={color:orange}> MOMO </h1>
    <h2> Model Optimization & Meta-learning Optimization </h2>
    <h3> A toolkit for self-supervised fine-tuning and inference for project Big Brain by Scrypt</h3>
    <img src="https://i.pinimg.com/originals/30/d8/99/30d899232dfe254a407a954880f424e4.gif" width="700px">
    <br>
    <img src="https://img.shields.io/badge/Lang-Python-blue.svg?style=for-the-badge&logo=python">
    <img src="https://img.shields.io/github/stars/PhantHive/momo?style=for-the-badge">
</div>


## Description

This project is a Python toolkit for machine learning model training and inference. It uses Gradio for creating easy-to-use UI around models and PEFT for fine-tuning models.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Usage

> For Gradio

Initialize the Gradio space by creating a repository that contains the following files:
- `toolkit/gradio/app.py`
- `toolkit/gradio/requirements.txt`

For fine-tuning, a Medium T4 GPU is required. For inference, a T4 GPU is recommended.

> For PEFT

Run on Google Colab with GPU enabled (T4 recommended) the following file:
- `toolkit/peft/llama-7b-finetune-peftqlora.ipynb`

## Dataset

> For fine-tuning

Dataset should be in the following format:
```
instruction
input
output
text
```
Where text is the concatenation of input and output:
`'input'  ->: 'output'`

Recommended dataset size is 1000 samples at least.
Recommended dataset format is .csv
Recommended dataset use stop tokens such as </s> to indicate end of input and output.

### Acknowledgements

> Resources used in this project are listed below:
> - [Gradio](https://www.gradio.app/guides/getting-started-with-the-python-client)
> - [PEFT](https://towardsdatascience.com/fine-tune-your-own-llama-2-model-in-a-colab-notebook-df9823a04a32)

