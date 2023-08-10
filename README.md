# Everything-about-LLMs

1. [Getting started](##Getting started: Karpathy's nanoGPT)
2. [Fine-tuning](##Fine-tuning)
    1. [LoRA](###LoRA)
    2. [QLoRA](###QLoRA)
    3. [RLHF](###RLHF)
4. [Multimodal models](##Multimodal models)
    1.[CLIP](###CLIP) 
3. [Engineering magics of training an LLM](##Engineering magics of training an LLM)

## Getting started: Karpathy's nanoGPT
[This folder](./karpathys_gpt) contains Karpathy's implementation of a mini version of GPT. 
You can run it to train a character-level language model on your laptop to generate shakespearean (well kind of :see_no_evil:) text.
He did a very nice tutorial to walk through the code almost line by line. 
You can watch it [here](https://www.youtube.com/watch?v=kCc8FmEb1nY).
If you are completely new to language modelling, [this video](https://www.youtube.com/watch?v=PaCmpygFfXo) may help you to understand more basics.

You can find much more details about the code in [Karpathy's original repo](https://github.com/karpathy/nanoGPT/tree/master#install). 
The code in this folder has been adapted to contain the minimal running code. 


## Fine-tuning
### LoRA

If you don't know what LoRA is, you can watch this Toutube video [here](https://www.youtube.com/watch?v=dA-NhCtrrVE) or
read the paper first: 

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2021,
Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen

I wrote a notebook to show how to fine-tune a **reeeeaaaal** simple binary classification model with LoRA, see [here](./LoRA.ipynb).

Of course, LoRA is already implemented for larger models as a library.  Here's another notebook [here](./LoRA_for_LLMs.ipynb) on how to fine-tune LLaMA 2 with the LoRA library.

### QLoRA

### RLHF

## Multimodal models

### CLIP

## Engineering magics of training an LLM
