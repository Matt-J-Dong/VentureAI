#!/bin/bash
#SBATCH --nodes=1                        # requests 1 compute server
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=4                # uses 4 compute cores per task
#SBATCH --time=48:00:00
#SBATCH --mem=258GB
#SBATCH --job-name=falcon8_model
#SBATCH --output=falcon8_model_copy.out
#SBATCH --error=falcon8_model-error_copy.txt
#SBATCH --mail-user=dq2024@nyu.edu
#SBATCH --mail-type=ALL

singularity exec --overlay /scratch/$USER/my_env/overlay-15GB-500K.ext3:rw /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash 
"
cd /scratch/dq2024/VentureAI
source /ext3/env.sh
cd /scratch/dq2024/VentureAI/fine_tuning

# Verify environment
pwd

# Check GPU status
nvidia-smi

# Source .env if it exists
if [ -f .env ]; then
    source .env
fi

# Set and export WANDB API key
# export WANDB_API_KEY=\$WANDB_API_KEY
# wandb login --relogin \$WANDB_API_KEY

# Run your script
./falcon8_model.sh
"