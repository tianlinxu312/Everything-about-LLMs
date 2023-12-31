#### Run the code
Install the dependencies:
```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

(copying the following **yada yaya yada** from Karpathy's repo...)

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

To train the model on a MacBook: 
```
python kaparthys_gpt/train.py --dataset shakespeare_char --batch_size 32 --lr 0.0001 --n_epochs 10 --n_workers 0
```

To generate text from your trained model:
```
python kaparthys_gpt/sample.py --out_dir=out-shakespeare-char --device=cpu
```