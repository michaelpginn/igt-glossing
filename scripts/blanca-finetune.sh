#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:v100
#SBATCH --ntasks=4          # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=48:00:00          # Max walltime              # Specify QOS
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
cd "/projects/migi8081/taxo-morph/src"

python3 finetune_token_classifier.py train --type flat --seed 1
python3 finetune_token_classifier.py train --type tax --seed 1
python3 finetune_token_classifier.py train --type tax-harmonic --seed 1

#for seed in 42 43 44 45 46 47 48 49 50 51
#do
#  for size in 10 100 500 1000
#  do
#    python3 finetune_token_classifier.py train flat --train_size $size --multitask false --seed $seed
#    python3 finetune_token_classifier.py train flat --train_size $size --multitask true --seed $seed
#    python3 finetune_token_classifier.py train --train_size $size --type multistage --seed $seed
#  done
#  python3 finetune_token_classifier.py train flat --train_size 10 --loss_sum linear --seed $seed
#  python3 finetune_token_classifier.py train tax --train_size 10 --loss_sum linear --seed $seed
#  python3 finetune_token_classifier.py train tax --train_size 10 --loss_sum harmonic --seed $seed
#done
