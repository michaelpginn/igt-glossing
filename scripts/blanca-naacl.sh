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

#python3 train_model.py train --model_type flat --seed 1  --train_data ../data/usp-train-track2-uncovered --eval_data ../data/usp-dev-track2-uncovered --test_data ../data/usp-test-track2-uncovered --project taxo-morph-naacl --epochs 200
for seed in 1 2 3 4 5
do
  python3 train_token_class_no_pretrained.py train --model_type flat --arch xlm-roberta --seed $seed --project taxo-morph-naacl --epochs 200
  python3 train_token_class_no_pretrained.py train --model_type tax_loss --arch xlm-roberta --seed $seed --project taxo-morph-naacl --epochs 200
  python3 train_token_class_no_pretrained.py train --model_type harmonic_loss --arch xlm-roberta --seed $seed --project taxo-morph-naacl --epochs 200
done