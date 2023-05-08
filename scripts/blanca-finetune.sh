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

for seed in 42 43 44 45 46 47 48 49 50 51
do
  for size in 10 100 500 1000
  do
#    for loss in tax tax_simple
#    do
    python3 finetune_token_classifier.py train tax --train_size $size --loss_sum harmonic --seed $seed
#    done
  done
done
