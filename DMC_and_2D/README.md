# Code for DeepMind Control Suite and 2D Env experiments 

## Code structure
The code is structured in the following way:
- `agents`:
    |_ `cmcgs.py`: we define the cmcgs agent in this file
- `configs`: all configs used in the method are defined here
- `toy_envs`: includes two customed environments: `two_d_navigation` and `two_d_reacher` for understanding the behavior and properties of the CMCGS.
- `utils`: helper functions are defined under this folder
- `wrappers`: environment wrappers

## How to run the code
1. We use `wandb` for logging, if you haven't used it before, login with
```shell
wandb login
```

2. Run the code by
```shell
python3 run_cmcgs.py use_wandb=true
```

3. If you want to run all the experiments, use the file `run_cmcgs_multiple.py` and the slurm jobs file `slurm-mpc.sh`
