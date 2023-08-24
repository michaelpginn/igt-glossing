#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:v100
#SBATCH --ntasks=4          # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=12:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-kann
#SBATCH --out=train_genbench_train_denoiser.%j.out      # Output file name
#SBATCH --error=train_genbench_train_denoiser.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
# Load the python module
module load anaconda
# Run Python Script
conda activate AutoIGT
cd "/projects/migi8081/taxo-morph/src"
python3 denoise_gloss_model.py --arch_size micro --project genbench-taxo-morph-denoiser --train_data ../data/GenBench/story_advice_personal --eval_data ../data/GenBench/history
