# Finetune example

KNN is all you need.

## Goals

Main
- train MLP for image classification
- tune the ResnetEncoder layers
- make embeddigns predictions and create the knn indedx model

Additional
- visualize embeddings with TF.Projector support
- find best starting lr with LRFinder support
- plot grid search metrics and compare them

### Preparation

Get the [data](https://www.dropbox.com/s/9438wx9ku9ke1pt/ants_bees.tar.gz) 
and unpack it to `data` folder.

Process the data
```bash
PYTHONPATH=. python prometheus/scripts/tag2label.py \
    --in-dir=./data/ants_bees \
    --out-dataset=./data/ants_bees/dataset.csv \
    --out-labeling=./data/ants_bees/tag2cls.json
```

And `pip install tensorflow` for visualization.

### Model training


```bash
export LOGDIR=$(pwd)/logs/finetune/baseline
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/src/ -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "PYTHONPATH=." \
   -e "LOGDIR=/logdir" \
   bite-gpu bash finetune/run_model.sh
```

### Tensorboard metrics visualization

For tensorboard support use 

`tensorboard --logdir=./logs/finetune`


### Embeddings projecting

```bash
export LOGDIR=$(pwd)/logs/finetune/baseline
bash finetune/run_projector.sh
tensorboard --logdir=./logs/finetune/projector
```

### Index model

```bash
export LOGDIR=$(pwd)/logs/finetune/baseline
docker run -it --rm --shm-size 8G \
   -v $(pwd):/src/ -v $LOGDIR:/logdir/ \
   -e "PYTHONPATH=." \
   -e "LOGDIR=/logdir" \
   bite-gpu bash finetune/run_index.sh
```

### LrFinder example

```bash
export LOGDIR=$(pwd)/logs/finetune/lrfinder
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/src/ -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "PYTHONPATH=." \
   -e "LOGDIR=/logdir" \
   bite-gpu bash finetune/run_lrfinder.sh
tensorboard --logdir=./logs/finetune/lrfinder
```

### Grid search metrics visualization

```bash
export BASELOGDIR=$(pwd)/logs/finetune
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/src/ -v $BASELOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "PYTHONPATH=." \
   -e "BASELOGDIR=/logdir" \
   bite-gpu bash finetune/run_grid.sh
tensorboard --logdir=./logs/finetune
```
