#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:v100
#SBATCH --ntasks=4          # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=12:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-kann
#SBATCH --out=train_genbench_exp1.%j.out      # Output file name
#SBATCH --error=train_genbench_exp1.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
# Load the python module
module load anaconda
# Run Python Script
conda activate AutoIGT
cd "/projects/migi8081/taxo-morph/src"

python3 finetune_token_classifier.py train --project genbench-taxo-morph-exp1 --model_type flat --seed 1 --epochs 50 --weight_decay 0 --train_data ../data/GenBench/train --eval_data ../data/GenBench/eval_OOD
python3 finetune_token_classifier.py train --project genbench-taxo-morph-exp1 --model_type flat --seed 1 --epochs 50 --weight_decay 0.01 --train_data ../data/GenBench/train --eval_data ../data/GenBench/eval_OOD
python3 finetune_token_classifier.py train --project genbench-taxo-morph-exp1 --model_type flat --seed 1 --epochs 50 --weight_decay 0.1 --train_data ../data/GenBench/train --eval_data ../data/GenBench/eval_OOD