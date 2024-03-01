#!/bin/bash

# Job name
#SBATCH -J "dvc repro"

# Time limit
#SBATCH -t 24:00:00

# Job resources
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32G
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1

# Notify by mail
#SBATCH --mail-type ALL
#SBATCH --mail-user christoph.berganski@uni-paderborn.de

# Load the python module
module load lang/Python/3.10.4-GCCcore-11.3.0

# Set up a new python environment in local memory
python3.10 -m venv /dev/shm/env/
# Load the python environment
source /dev/shm/env/bin/activate
# Install required packages
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# Reproduce the training and evaluation pipeline
dvc repro
