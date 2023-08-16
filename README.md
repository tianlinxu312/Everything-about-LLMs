# Everything-about-LLMs

1. [Getting started](#Getting-started-Karpathys-nanoGPT)
2. [Fine-tuning](#Fine-tuning)
    1. [LoRA](#LoRA)
    2. [QLoRA](#QLoRA)
    3. [RLHF](#RLHF)
4. [Multimodal models](#Multimodal-models)
    1. [CLIP](#CLIP) 
3. [Engineering magics of training an LLM](#Engineering-magics-of-training-an-LLM)

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

If you don't know what LoRA is, you can watch this Toutube video [here](https://www.youtube.com/watch?v=dA-NhCtrrVE), or
read the [LoRA paper](https://arxiv.org/abs/2106.09685)[^1] first. 

- Toy problem: I wrote a notebook to show how to fine-tune a **reeeeaaaal** simple binary classification model with LoRA, see [here](./LoRA.ipynb).

 - The real deal: of course, some amazing people already implemented LoRA as a library.  Here's [the notebook](./LoRA_for_LLMs.ipynb) on how to fine-tune LLaMA 2 with the LoRA library.

### QLoRA

As discussed in the [LoRA for LLMs notebook](./LoRA_for_LLMs.ipynb), we only need to train about 12% of the original parameter count by applying this low rank representation.  However, we still have to load the entire model, as the low rank weight matrix is added to the orginal weights. For the smallest Llama 2 model with 7 billion parameters, it will require 28G memory on the GPU allocated just to store the parameters, making it impossible to train on lower-end GPUs such as T4 or V100.  

Therefore, (**...drum rolls...**) [QLoRA](https://arxiv.org/pdf/2305.14314.pdf)[^2] was proposed.  QLoRA loads the 4-bit quantized weights from a pretrained model, and then apply LoRA to fine tune the model.  There are more technical details you may be interested in. If so, you can read the paper or watch this video [here](https://www.youtube.com/watch?v=TPcXVJ1VSRI). 

With the LoRA library (check the notebook), it is very easy to adopt QLoRA.  All you need to do is to specify in the configuration as below: 
```
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=True,
                                             load_in_4bit=True, # <------ *here*
                                             #  load_in_8bit=True,
                                             )
```
Of course, quantization leads to information loss.  This is a tradeoff between memory and accuracy.  If needed, there's also an 8-bit option. 

As a result, we can fine tune a 7-billon-param model on a single T4 GPU.  Check out the RAM usage during training: 

![image](./imgs/GPU_usage.png)


### RLHF

## Multimodal models

### CLIP

### DALL·E 2

### Stable Diffusion

## Engineering magics of training an LLM

### Memory Optimization: ZeRO

### Model parallelism: MegatronLM

### Pipeline Parallelism

### Checkpointing and Deterministic Training  

### FlashAttention

## KV caching

## Reference:

[^1]: Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. and Chen, W., 2021. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), arXiv preprint arXiv:2106.09685

[^2]: Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L., 2023. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf). arXiv preprint arXiv:2305.14314.
