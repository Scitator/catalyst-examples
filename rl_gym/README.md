# Catalyst.RL – OpenAI Gym LunarLander example

@TODO: find appropriate hyperparameters for OpenAI Gym Envs


1. System requirements – redis

    `sudo apt install redis-server`

2. Python requirements

    ```bash
    pip install torch torchvision git+https://github.com/pytorch/tnt.git@master
    pip install pyaml tensorboardX jpeg4py albumentations
    pip isntall redis gym gym['Box2D']
    ```

3. Run

    ```bash
    redis-server --port 12000
    export GPUS=""
    CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. \
        python catalyst/rl/offpolicy/scripts/run_trainer.py \
        --config=./rl_gym/config.yml
    
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        python catalyst/rl/offpolicy/scripts/run_samplers.py \
        --config=./rl_gym/config.yml
    
    CUDA_VISIBLE_DEVICE="" tensorboard \
       --logdir=./logs/rl_gym
    ```


## Additional links

[NeurIPS'18 Catalyst.RL solution](https://github.com/Scitator/neurips-18-prosthetics-challenge)

