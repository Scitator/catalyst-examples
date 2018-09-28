#!/usr/bin/env bash
set -e

echo "index model creating..."
PYTHONPATH=. python prometheus/scripts/create_index_model.py \
   --in-npy=${LOGDIR}/dataset.predictions.infer.embeddings.npy \
   --n-hidden=16 --knn-metric="l2" \
   --out-npy=${LOGDIR}/dataset.predictions.infer.embeddings.pca.npy \
   --out-pipeline=${LOGDIR}/pipeline.embeddings.pkl \
   --out-knn=${LOGDIR}/knn.embeddings.bin

echo "index model testing..."
PYTHONPATH=. python prometheus/scripts/check_index_model.py \
   --in-npy=${LOGDIR}/dataset.predictions.infer.embeddings.pca.npy \
   --in-knn=${LOGDIR}/knn.embeddings.bin \
   --knn-metric="l2" --batch-size=64 | tee ${LOGDIR}/index_check.txt

# docker trick
if [ "$EUID" -eq 0 ]; then
  chmod -R 777 ${LOGDIR}
fi
