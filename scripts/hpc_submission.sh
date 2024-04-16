#!/bin/sh

### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J PI-IW-Atari
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 1GB of memory per core/slot --
#BSUB -R "rusage[mem=1GB]"
### -- specify that we want the job to get killed if it exceeds 2 GB per core/slot --
#BSUB -M 1GB
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
python3 ../piiw/online_planning_learning_lightning.py --config-name config_atari_dynamic.yaml > output.out