#!/usr/bin/env bash
set -e

echo "Training...1"
PYTHONPATH=. python catalyst/dl/scripts/train.py \
    --model-dir=finetune \
    --config=finetune/train.yml \
    --baselogdir=${BASELOGDIR} --verbose

echo "Training...2"
PYTHONPATH=. python catalyst/dl/scripts/train.py \
    --model-dir=finetune \
    --config=finetune/train.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --model_params/img_encoder/pooling=GlobalAvgPool2d:str \
    --model_params/cls_net/hiddens=[512]:list

echo "Training...3"
PYTHONPATH=. python catalyst/dl/scripts/train.py \
    --model-dir=finetune \
    --config=finetune/train.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --model_params/img_encoder/pooling=GlobalMaxPool2d:str \
    --model_params/cls_net/hiddens=[512]:list

echo "Training...4"
PYTHONPATH=. python catalyst/dl/scripts/train.py \
    --model-dir=finetune \
    --config=finetune/train.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --model_params/cls_net/emb_size=128:int

# docker trick
if [ "$EUID" -eq 0 ]; then
  chmod -R 777 ${BASELOGDIR}
fi
