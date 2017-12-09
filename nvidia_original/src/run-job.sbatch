#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=eyeglasss
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=ag4508@nyu.edu

module load python/intel/2.7.12
module load pytorch/0.2.0_1
module load protobuf/intel/3.1.0
module load tensorboard_logger/0.0.3
module load torchvision/0.1.8
module load opencv/intel/3.2
pip install --user tensorboard
#source activate nlp
python cocogan_train.py --config ../exps/unit/anant_eyeglasses.yaml --log /scratch/ag4508/unit/log/ 
