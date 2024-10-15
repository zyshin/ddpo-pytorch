. /mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/yshzhu/bashrc

# set -xe
cd $(dirname $(realpath $0))

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

accelerate launch scripts/train.py --config config/dgx.py:aesthetic2
# accelerate launch scripts/train.py --config config/dgx.py:intrinsic_only
# accelerate launch scripts/train.py --config config/dgx.py:new_baseline_aesthetic
# accelerate launch scripts/train.py --config config/dgx.py:compressibility
# mgpu_active
