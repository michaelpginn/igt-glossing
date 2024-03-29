#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00          # Max walltime
#SBATCH --out=train_genbench_exp4.%j.out      # Output file name
#SBATCH --error=train_genbench_exp4.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# Load the python module
module load gcc/9.3.0 arrow/8
module load python/3.10
module load scipy-stack

# Create the env
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install -r ~/scratch/taxo-morph/requirements.txt

# Run Python Script
cd ~/scratch/taxo-morph/src


python3 finetune_token_classifier.py train --project genbench-taxo-morph-exp1 --model_type multistage --seed 1 --epochs 50 --weight_decay 0.75 --train_data ../data/GenBench/train.txt --eval_data ../data/GenBench/eval_ood.txt
