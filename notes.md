```
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
cog run scripts/prepare.sh $MODEL_REPO <hf-token here>
cog run python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"
```