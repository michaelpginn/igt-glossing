#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:v100
#SBATCH --ntasks=4          # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=12:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-kann
#SBATCH --out=train_structmorph.%j.out      # Output file name
#SBATCH --error=train_structmorph.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
# Load the python module
module load anaconda
# Run Python Script
conda activate AutoIGT
cd "/projects/migi8081/struct-morph/src"
python3 finetune_token_classifier.py train tax --train_size 10 --seed 42
python3 finetune_token_classifier.py train tax --train_size 100 --seed 42
python3 finetune_token_classifier.py train tax --train_size 500 --seed 42
python3 finetune_token_classifier.py train tax --train_size 1000 --seed 42
python3 finetune_token_classifier.py train tax --seed 42
