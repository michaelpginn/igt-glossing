#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:v100
#SBATCH --ntasks=4          # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=1:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-kann
#SBATCH --out=train_genbench_pretrain.%j.out      # Output file name
#SBATCH --error=train_genbench_pretrain.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
# Load the python module
module load anaconda
# Run Python Script
conda activate AutoIGT
cd "/projects/migi8081/taxo-morph/src"
python3 pretrain_language_model.py --arch_size micro --project genbench-taxo-morph-pretrain --train_data ../data/GenBench/train.txt --eval_data ../data/GenBench/eval_id.txt --position_embeddings absolute
#python3 pretrain_language_model.py --arch_size full --project genbench-taxo-morph-pretrain --train_data ../data/GenBench/train --eval_data ../data/GenBench/eval_ID --position_embeddings relative_key_query
