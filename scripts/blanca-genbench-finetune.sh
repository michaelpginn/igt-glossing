#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:v100
#SBATCH --ntasks=4          # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=48:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-kann
#SBATCH --out=genbench_finetune.%j.out      # Output file name
#SBATCH --error=genbench_finetune.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
# Load the python module
module load anaconda
# Run Python Script
conda activate AutoIGT
cd "/projects/migi8081/taxo-morph/src"

for seed in 1 2 3 4 5
do
  for type in flat tax_loss harmonic_loss multitask multistage
  do
    python3 finetune_token_classifier.py train --model_type $type --seed $seed --train_data ../data/GenBench/story --eval_data ../data/GenBench/nonstory
  done
done

for size in 10 100 500 1000
do
  for seed in 1 2 3 4 5
  do
    for type in flat tax_loss harmonic_loss multitask multistage
    do
      python3 finetune_token_classifier.py train --model_type $type --train_size $size --seed $seed --train_data ../data/GenBench/story --eval_data ../data/GenBench/nonstory
    done
  done
done