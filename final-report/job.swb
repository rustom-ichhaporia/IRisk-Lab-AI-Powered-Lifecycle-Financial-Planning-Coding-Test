#!/bin/bash
#SBATCH --job-name="script"
#SBATCH --output="script.%j.%N.out"
#SBATCH --partition=cpun1

module load wmlce
pip install imblearn
pip install lightgbm==2.3.0
pip install hyperopt==0.2.5
python script.py