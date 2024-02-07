#!/bin/bash
#SBATCH --cpus-per-task 10
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu=3000
#SBATCH --array=0-119

case $SLURM_ARRAY_TASK_ID in
    0) ARGS="env_name=walker-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=5000" ;;
    1) ARGS="env_name=walker-walk save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=5000" ;;
    2) ARGS="env_name=reacher-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=5000" ;;
    3) ARGS="env_name=finger-spin save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=5000" ;;
    4) ARGS="env_name=cheetah-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=5000" ;;
    5) ARGS="env_name=cartpole-swingup save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=5000" ;;
    6) ARGS="env_name=cartpole-balance_sparse save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=5000" ;;
    7) ARGS="env_name=ball_in_cup-catch save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=5000" ;;
    8) ARGS="env_name=2d-reacher-thirty-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=5000" ;;
    9) ARGS="env_name=2d-reacher-fifteen-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=5000" ;;
    10) ARGS="env_name=2d-navigation-boxes save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=5000" ;;
    11) ARGS="env_name=2d-navigation-circles save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=5000" ;;
    12) ARGS="env_name=walker-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4500" ;;
    13) ARGS="env_name=walker-walk save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4500" ;;
    14) ARGS="env_name=reacher-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4500" ;;
    15) ARGS="env_name=finger-spin save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4500" ;;
    16) ARGS="env_name=cheetah-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4500" ;;
    17) ARGS="env_name=cartpole-swingup save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4500" ;;
    18) ARGS="env_name=cartpole-balance_sparse save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4500" ;;
    19) ARGS="env_name=ball_in_cup-catch save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4500" ;;
    20) ARGS="env_name=2d-reacher-thirty-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4500" ;;
    21) ARGS="env_name=2d-reacher-fifteen-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4500" ;;
    22) ARGS="env_name=2d-navigation-boxes save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4500" ;;
    23) ARGS="env_name=2d-navigation-circles save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4500" ;;
    24) ARGS="env_name=walker-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4000" ;;
    25) ARGS="env_name=walker-walk save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4000" ;;
    26) ARGS="env_name=reacher-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4000" ;;
    27) ARGS="env_name=finger-spin save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4000" ;;
    28) ARGS="env_name=cheetah-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4000" ;;
    29) ARGS="env_name=cartpole-swingup save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4000" ;;
    30) ARGS="env_name=cartpole-balance_sparse save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4000" ;;
    31) ARGS="env_name=ball_in_cup-catch save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4000" ;;
    32) ARGS="env_name=2d-reacher-thirty-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4000" ;;
    33) ARGS="env_name=2d-reacher-fifteen-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4000" ;;
    34) ARGS="env_name=2d-navigation-boxes save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4000" ;;
    35) ARGS="env_name=2d-navigation-circles save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=4000" ;;
    36) ARGS="env_name=walker-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3500" ;;
    37) ARGS="env_name=walker-walk save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3500" ;;
    38) ARGS="env_name=reacher-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3500" ;;
    39) ARGS="env_name=finger-spin save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3500" ;;
    40) ARGS="env_name=cheetah-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3500" ;;
    41) ARGS="env_name=cartpole-swingup save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3500" ;;
    42) ARGS="env_name=cartpole-balance_sparse save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3500" ;;
    43) ARGS="env_name=ball_in_cup-catch save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3500" ;;
    44) ARGS="env_name=2d-reacher-thirty-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3500" ;;
    45) ARGS="env_name=2d-reacher-fifteen-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3500" ;;
    46) ARGS="env_name=2d-navigation-boxes save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3500" ;;
    47) ARGS="env_name=2d-navigation-circles save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3500" ;;
    48) ARGS="env_name=walker-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3000" ;;
    49) ARGS="env_name=walker-walk save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3000" ;;
    50) ARGS="env_name=reacher-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3000" ;;
    51) ARGS="env_name=finger-spin save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3000" ;;
    52) ARGS="env_name=cheetah-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3000" ;;
    53) ARGS="env_name=cartpole-swingup save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3000" ;;
    54) ARGS="env_name=cartpole-balance_sparse save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3000" ;;
    55) ARGS="env_name=ball_in_cup-catch save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3000" ;;
    56) ARGS="env_name=2d-reacher-thirty-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3000" ;;
    57) ARGS="env_name=2d-reacher-fifteen-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3000" ;;
    58) ARGS="env_name=2d-navigation-boxes save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3000" ;;
    59) ARGS="env_name=2d-navigation-circles save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=3000" ;;
    60) ARGS="env_name=walker-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2500" ;;
    61) ARGS="env_name=walker-walk save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2500" ;;
    62) ARGS="env_name=reacher-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2500" ;;
    63) ARGS="env_name=finger-spin save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2500" ;;
    64) ARGS="env_name=cheetah-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2500" ;;
    65) ARGS="env_name=cartpole-swingup save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2500" ;;
    66) ARGS="env_name=cartpole-balance_sparse save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2500" ;;
    67) ARGS="env_name=ball_in_cup-catch save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2500" ;;
    68) ARGS="env_name=2d-reacher-thirty-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2500" ;;
    69) ARGS="env_name=2d-reacher-fifteen-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2500" ;;
    70) ARGS="env_name=2d-navigation-boxes save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2500" ;;
    71) ARGS="env_name=2d-navigation-circles save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2500" ;;
    72) ARGS="env_name=walker-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2000" ;;
    73) ARGS="env_name=walker-walk save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2000" ;;
    74) ARGS="env_name=reacher-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2000" ;;
    75) ARGS="env_name=finger-spin save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2000" ;;
    76) ARGS="env_name=cheetah-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2000" ;;
    77) ARGS="env_name=cartpole-swingup save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2000" ;;
    78) ARGS="env_name=cartpole-balance_sparse save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2000" ;;
    79) ARGS="env_name=ball_in_cup-catch save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2000" ;;
    80) ARGS="env_name=2d-reacher-thirty-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2000" ;;
    81) ARGS="env_name=2d-reacher-fifteen-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2000" ;;
    82) ARGS="env_name=2d-navigation-boxes save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2000" ;;
    83) ARGS="env_name=2d-navigation-circles save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=2000" ;;
    84) ARGS="env_name=walker-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1500" ;;
    85) ARGS="env_name=walker-walk save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1500" ;;
    86) ARGS="env_name=reacher-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1500" ;;
    87) ARGS="env_name=finger-spin save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1500" ;;
    88) ARGS="env_name=cheetah-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1500" ;;
    89) ARGS="env_name=cartpole-swingup save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1500" ;;
    90) ARGS="env_name=cartpole-balance_sparse save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1500" ;;
    91) ARGS="env_name=ball_in_cup-catch save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1500" ;;
    92) ARGS="env_name=2d-reacher-thirty-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1500" ;;
    93) ARGS="env_name=2d-reacher-fifteen-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1500" ;;
    94) ARGS="env_name=2d-navigation-boxes save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1500" ;;
    95) ARGS="env_name=2d-navigation-circles save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1500" ;;
    96) ARGS="env_name=walker-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1000" ;;
    97) ARGS="env_name=walker-walk save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1000" ;;
    98) ARGS="env_name=reacher-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1000" ;;
    99) ARGS="env_name=finger-spin save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1000" ;;
    100) ARGS="env_name=cheetah-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1000" ;;
    101) ARGS="env_name=cartpole-swingup save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1000" ;;
    102) ARGS="env_name=cartpole-balance_sparse save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1000" ;;
    103) ARGS="env_name=ball_in_cup-catch save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1000" ;;
    104) ARGS="env_name=2d-reacher-thirty-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1000" ;;
    105) ARGS="env_name=2d-reacher-fifteen-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1000" ;;
    106) ARGS="env_name=2d-navigation-boxes save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1000" ;;
    107) ARGS="env_name=2d-navigation-circles save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=1000" ;;
    108) ARGS="env_name=walker-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=500" ;;
    109) ARGS="env_name=walker-walk save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=500" ;;
    110) ARGS="env_name=reacher-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=500" ;;
    111) ARGS="env_name=finger-spin save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=500" ;;
    112) ARGS="env_name=cheetah-run save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=500" ;;
    113) ARGS="env_name=cartpole-swingup save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=500" ;;
    114) ARGS="env_name=cartpole-balance_sparse save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=500" ;;
    115) ARGS="env_name=ball_in_cup-catch save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=500" ;;
    116) ARGS="env_name=2d-reacher-thirty-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=500" ;;
    117) ARGS="env_name=2d-reacher-fifteen-poles-hard save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=500" ;;
    118) ARGS="env_name=2d-navigation-boxes save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=500" ;;
    119) ARGS="env_name=2d-navigation-circles save_video=false use_wandb=false agent=cmcgs agent.params.simulation_budget=500" ;;
esac

# FILL IN YOUR OWN ACTIVATIONS HERE
conda activate cmcgs

# RUN THE EXPERIMENT
srun python run_cmcgs_multiple.py $ARGS
