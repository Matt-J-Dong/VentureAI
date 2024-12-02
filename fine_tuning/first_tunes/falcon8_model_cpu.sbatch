#!/bin/bash
#SBATCH --nodes=1                        # requests 1 compute server
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=4                # uses 4 compute cores per task
#SBATCH --time=48:00:00
#SBATCH --mem=258GB
#SBATCH --job-name=falcon8_model_cpu
#SBATCH --output=falcon8_model_cpu.out
#SBATCH --error=falcon8_model_cpu-error.txt
#SBATCH --mail-user=matthew.dong@nyu.edu
#SBATCH --mail-type=ALL

singularity exec --nv --overlay $SCRATCH/new_env/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
cd $SCRATCH/VentureAI
source myenv/bin/activate
cd $SCRATCH/VentureAI/fine_tuning

# Verify environment
pwd
env_list=\$(pip list)
echo "\$env_list"

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