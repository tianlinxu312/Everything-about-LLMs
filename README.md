# Everything-about-LLMs

## Kaparthy's nanoGPT
[This folder](./karpathys_gpt) contains Kaparthy's implementation of a mini version of GPT. 
You can run it to train a character-level language model to generate shakespearean (well kind of) text on your laptop.
He did a very nice tutorial to walk through the code almost line by line. 
You can watch it [here](https://www.youtube.com/watch?v=kCc8FmEb1nY).
If you are completely new to language modelling, this [video](https://www.youtube.com/watch?v=PaCmpygFfXo) may help you to understand more basics.

You can also find much more details about the code in [Kaparthy's original repo](https://github.com/karpathy/nanoGPT/tree/master#install). This folder here only contains the minimal running code. 

#### Run the code
Copying the dependencies needed to run this code over:
```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

(copying the following **yada yaya yada**...)

--------------------------------
Dependencies:
- pytorch <3
- numpy <3
- transformers for huggingface transformers <3 (to load GPT-2 checkpoints)
- datasets for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
- tiktoken for OpenAI's fast BPE code <3
- wandb for optional logging <3
- tqdm for progress bars <3
--------------------------------

Prepare the data:
```
python kaparthys_gpt/data/shakespeare_char/prepare.py
```

To train in the model on a MacBook: 
```
python kaparthys_gpt/train.py --dataset shakespeare_char --batch_size 32 --lr 0.0001 --n_epochs 10 --n_workers 0
```

To generate text from your trained model:
```
python kaparthys_gpt/sample.py --out_dir=out-shakespeare-char --device=cpu
```

## Fine-tuning
### LoRA

### QLoRA

### RLHF
