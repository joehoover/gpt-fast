hf_token=${2:-None}
python scripts/download.py --repo_id $1 --hf_token $hf_token && python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$1 && python quantize.py --checkpoint_path checkpoints/$1/model.pth --mode int8
