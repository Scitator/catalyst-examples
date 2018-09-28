#!/usr/bin/env bash
set -e

PYTHONPATH=. python prometheus/dl/scripts/train.py \
    --model-dir=finetune \
    --config=finetune/debug.yml \
    --logdir=${LOGDIR} --verbose

# docker trick
if [ "$EUID" -eq 0 ]; then
  chmod -R 777 ${LOGDIR}
fi
