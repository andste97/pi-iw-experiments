##!/bin/sh

### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J PI-IW-gpu
### -- ask for number of cores (default: 1) --
#BSUB -n 13
### -- Select the resources: 1 gpu in shared mode --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 20GB of memory --
#BSUB -R "rusage[mem=2300MB]"
### -- specify that we want the job to get killed if it exceeds 30GB --
#BSUB -M 2500MB
### -- set walltime limit: hh:mm --
#BSUB -W 24:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s220278@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o hpc_out/Output_%J.out
#BSUB -e hpc_out/Output_%J.err

# here follow the commands you want to execute with input.in as the input file
# shellcheck disable=SC2039
source ../venv/bin/activate
#wandb offline
python3 ../piiw/online_planning_learning_run_entire_suite.py --config-name=config_atari_dynamic.yaml train.learning_rate=0.0005 train.total_interaction_budget=20000000 plan.softmax_temperature=0.5 plan.risk_averse=True plan.softmax_decay=True 'env_suite=["Pong-v4", "MsPacman-v4", "ChopperCommand-v4", "Breakout-v4"]' > output.out
#wandb online
