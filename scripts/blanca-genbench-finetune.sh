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


#python3 finetune_token_classifier.py train --project genbench-taxo-morph-finetune-final --model_type flat --seed 1 --train_data ../data/GenBench/story --eval_data ../data/GenBench/eval_ID --epochs 100
#python3 finetune_token_classifier.py train --project genbench-taxo-morph-finetune-final --model_type harmonic_loss --seed 1 --train_data ../data/GenBench/story --eval_data ../data/GenBench/eval_ID --epochs 100
python3 finetune_token_classifier.py train --project genbench-taxo-morph-finetune-final --model_type denoised --seed 1 --train_data ../data/GenBench/story --eval_data ../data/GenBench/eval_ID --epochs 100
