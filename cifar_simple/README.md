## How to run

### Local run

```bash
PYTHONPATH=. python prometheus/dl/scripts/train.py \
   --config=./cifar_simple/config.yml
```

For tensorboard support use 

`tensorboard --logdir=./logs/cifar_simple`


### Docker run

For docker image goto `prometheus/docker`

```bash
export LOGDIR=$(pwd)/logs/cifar_simple_docker
docker run -it --rm \
   -v $(pwd):/src -v $LOGDIR:/logdir/ \
   -e PYTHONPATH=. \
   pro-cpu python prometheus/dl/scripts/train.py \
   --config=./cifar_simple/config.yml --logdir=/logdir
```


For tensorboard support use 

`tensorboard --logdir=./logs/cifar_docker`