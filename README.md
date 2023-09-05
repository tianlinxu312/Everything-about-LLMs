# Everything-about-LLMs

1. [Getting started](#Getting-started-Karpathys-nanoGPT)
2. [Fine-tuning](#Fine-tuning)
    1. [LoRA](#LoRA)
    2. [QLoRA](#QLoRA)
    3. [RLHF](#RLHF)
3. [Multimodal models](#Multimodal-models)
    1. [CLIP](#CLIP)
    2. [GLIDE](#GLIDE)
    3. [DALL·E 2](#DALL·E-2)
    4. [Stable Diffusion](#Stable-Diffusion)
        1. [Implementation: Image to Image](#Image-to-Image)
4. [Engineering magics for training an LLM](#Engineering-magics-for-training-an-LLM)
    1. [Memory Optimization: ZeRO](#Memory-Optimization-ZeRO)
    2. [Model parallelism: MegatronLM](#Model-parallelism-MegatronLM)
    3. [Pipeline Parallelism](#Pipeline-Parallelism)
    4. [Checkpointing and Deterministic Training](#Checkpointing-and-Deterministic-Training)
    5. [FlashAttention](#FlashAttention)
    6. [KV caching](#KV-caching)
    7. [Gradient checkpointing](#Gradient-checkpointing)
    8. [Data efficiency](#Data-efficiency)


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
read the LoRA paper[^1] first. 

- Toy problem: I wrote a notebook to show how to fine-tune a **reeeeaaaal** simple binary classification model with LoRA, see [here](./LoRA.ipynb).

 - The real deal: of course, some amazing people already implemented LoRA as a library.  Here's [the notebook](./LoRA_for_LLMs.ipynb) on how to fine-tune LLaMA 2 with the LoRA library.

### QLoRA

As discussed in the [LoRA for LLMs notebook](./LoRA_for_LLMs.ipynb), we only need to train about 12% of the original parameter count by applying this low rank representation.  However, we still have to load the entire model, as the low rank weight matrix is added to the orginal weights. For the smallest Llama 2 model with 7 billion parameters, it will require 28G memory on the GPU allocated just to store the parameters, making it impossible to train on lower-end GPUs such as T4 or V100.  

Therefore, (**...drum rolls...**) QLoRA[^2] was proposed.  QLoRA loads the 4-bit quantized weights from a pretrained model, and then apply LoRA to fine tune the model.  There are more technical details you may be interested in. If so, you can read the paper or watch this video [here](https://www.youtube.com/watch?v=TPcXVJ1VSRI). 

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
Unfortunately, quantization leads to an information loss.  This is a tradeoff between memory and accuracy.  If needed, there's also an 8-bit option. 

By choosing to load the entire pre-trained model in 4-bit, we can fine-tune a 7-billon-parameter model on a single T4 GPU.  Check out the RAM usage during training: 

![image](./imgs/GPU_usage.png)


### RLHF

## Multimodal models

### CLIP

Concenptually, CLIP is very simple. The figure in the CLIP paper[^3] says it all.  

![image](./imgs/clip.png)

For this visual-language application, step (1) in the figure needs a few components:
- data: images with text describing them
- a visual encoder to extract image features
- a language encoder to extract text features
- learn by maximising the similarity between the paired image and text features indicated by the blue squares in the matrix in the figure (contrastive learning)

I wrote a (very) simple example in [this notebook](./CLIP_for_MNIST.ipynb) which implements and explains the contrastive learning objective, and describes the components in step (2) and (3). However, I used the same style of text labels for training and testing.  So no zero-shot here.  

### GLIDE

GLIDE[^8] is a text-to-image diffusion model with CLIP as the guidance.  If you aren't familiar with diffusion models, you can watch [this video](https://www.youtube.com/watch?v=344w5h24-h8&list=PLpZBeKTZRGPPvAyM9DM-a6W0lugCo8WfC) for a quick explaination to the concept.  If you want more technical details, you can start with these papers: diffusion generative model[^5], DDPM[^6], DDIM[^7], and a variational perspective of diffusion models[^9].  

### DALL·E 2

DALL·E 2 is another concenptually simply model that produces amazing results.  

The first half of the model is a pre-trained CLIP (frozen once trained), i.e., the part above the dash line in the figure in the DALL·E 2 paper[^4], see below.  

![image](./imgs/dalle2.png)

In CLIP, we have trained two encoders to extract features from image and text inputs.  

### Stable Diffusion

#### Image to Image
An implementation of Stable Diffusion Image-to-Image can be found [here](./stable_diffusion_img2img.py). Alternatively, you can also play with [this notebook](https://colab.research.google.com/drive/15MS1tAK69Nbdv6cnkN0U8GrJAYwtbuw9?usp=sharing).  


## Engineering magics for training an LLM

### Memory Optimization: ZeRO

### Model parallelism: MegatronLM

### Pipeline Parallelism

### Checkpointing and Deterministic Training  

### FlashAttention

### KV caching

### Gradient checkpointing

### Data efficiency

## Reference:

[^1]: Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. and Chen, W., 2021. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), arXiv preprint arXiv:2106.09685

[^2]: Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L., 2023. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf). arXiv preprint arXiv:2305.14314.

[^3]: Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J. and Krueger, G., 2021, July. [Learning transferable visual models from natural language supervision](http://proceedings.mlr.press/v139/radford21a). In International conference on machine learning (pp. 8748-8763). PMLR.

[^4]: Ramesh, A., Dhariwal, P., Nichol, A., Chu, C. and Chen, M., 2022. [Hierarchical text-conditional image generation with clip latents](https://cdn.openai.com/papers/dall-e-2.pdf). arXiv preprint arXiv:2204.06125, 1(2), p.3.

[^5]: Sohl-Dickstein, Jascha; Weiss, Eric; Maheswaranathan, Niru; Ganguli, Surya (2015-06-01). [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](http://proceedings.mlr.press/v37/sohl-dickstein15.pdf). Proceedings of the 32nd International Conference on Machine Learning. PMLR. 37: 2256–2265

[^6]:Ho, J., Jain, A. and Abbeel, P., 2020. [Denoising diffusion probabilistic models](https://arxiv.org/pdf/2006.11239.pdf). Advances in neural information processing systems, 33, pp.6840-6851.

[^7]: Song, J., Meng, C. and Ermon, S., 2020. [Denoising diffusion implicit models](https://arxiv.org/pdf/2010.02502.pdf). arXiv preprint arXiv:2010.02502.

[^8]: Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I. and Chen, M., 2021. [Glide: Towards photorealistic image generation and editing with text-guided diffusion models](https://arxiv.org/pdf/2112.10741.pdf). arXiv preprint arXiv:2112.10741. 

[^9]: Kingma, D., Salimans, T., Poole, B. and Ho, J., 2021. [Variational diffusion models](https://proceedings.neurips.cc/paper/2021/hash/b578f2a52a0229873fefc2a4b06377fa-Abstract.html). Advances in neural information processing systems, 34, pp.21696-21707.
