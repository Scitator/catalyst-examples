#!/usr/bin/env bash
set -e

echo "Training...0"
PYTHONPATH=. python catalyst/dl/scripts/train.py \
    --model-dir=finetune \
    --config=finetune/train.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --stages/data_params/train_folds=[1,2,3,4]:list

echo "Training...1"
PYTHONPATH=. python catalyst/dl/scripts/train.py \
    --model-dir=finetune \
    --config=finetune/train.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --stages/data_params/train_folds=[0,2,3,4]:list

echo "Training...2"
PYTHONPATH=. python catalyst/dl/scripts/train.py \
    --model-dir=finetune \
    --config=finetune/train.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --stages/data_params/train_folds=[0,1,3,4]:list

echo "Training...3"
PYTHONPATH=. python catalyst/dl/scripts/train.py \
    --model-dir=finetune \
    --config=finetune/train.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --stages/data_params/train_folds=[0,1,2,4]:list

echo "Training...4"
PYTHONPATH=. python catalyst/dl/scripts/train.py \
    --model-dir=finetune \
    --config=finetune/train.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --stages/data_params/train_folds=[0,1,2,3]:list

# docker trick
if [ "$EUID" -eq 0 ]; then
  chmod -R 777 ${BASELOGDIR}
fi
