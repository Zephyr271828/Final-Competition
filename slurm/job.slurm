#!/bin/bash

#SBATCH --job-name=final_competition
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --cpus-per-task=14
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin        
#SBATCH --mail-type=end           
#SBATCH --mail-user=yx3038@nyu.edu
#SBATCH --requeue

singularity exec --nv --bind /scratch/yx3038 --overlay /scratch/yx3038/overlay-25GB-500K.ext3:ro /scratch/yx3038/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/yx3038/24Spring/ML/Final-Competition/src
python main.py --epochs=20 --input_size=256"
