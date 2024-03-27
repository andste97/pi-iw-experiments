#!/bin/sh

# Check if required parameters are input as flags
SWEEP_AGENT_ID=""
while getopts ":i:" opt; do
  case ${opt} in
    i )
      SWEEP_AGENT_ID=$OPTARG
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))

# Ensure the SWEEP_AGENT_ID is provided
if [ -z "$SWEEP_AGENT_ID" ]; then
  echo "Error: Sweep agent ID not provided. Use -i <ID> to specify it."
  exit 1
fi

### General options
### -- set the job Name AND the job array --
#BSUB -J PIIW-sweep[1-5]
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
#BSUB -W 48:00
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
#BSUB -o hpc_out/Output_%J_%I.out
#BSUB -e hpc_out/Output_%J_%I.err

# Activate the virtual environment
# shellcheck disable=SC2039
source ../venv/bin/activate

# go to project root
cd ..
# Start the wandb sweep agent with the provided ID
wandb agent piiw-thesis/piiw-sweep/"$SWEEP_AGENT_ID"
