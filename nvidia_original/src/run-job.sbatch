#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=hairsmile4wayBIG
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
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

#python cocogan_train.py --config ../exps/unit/anant_smiling.yaml --log /scratch/ag4508/unit/log/ 
#python cocogan_train.py --config ../exps/unit/anant_eyeglasses.yaml --log /scratch/ag4508/unit/log/ 
#python cocogan_train.py --config ../exps/unit/anant_goatee.yaml --log /scratch/ag4508/unit/log/ 
#python cocogan_train.py --config ../exps/unit/anant_male.yaml --log /scratch/ag4508/unit/log/ 
python cocogan_train_fourway.py --config ../exps/unit/anant_smiling_eyeglass_4wayjoint.yaml --log /scratch/ag4508/unit/log/ 
#python cocogan_train_fourway.py --config ../exps/unit/anant_smiling_eyeglass_bigK_warmstart.yaml --log /scratch/ag4508/unit/log/ --warm_start 1 --gen_ab /scratch/ag4508/unit/exps/eyeglass_bigK/eyeglass_bigK_gen_00200000.pkl --dis_ab  /scratch/ag4508/unit/exps/eyeglass_bigK/eyeglass_bigK_dis_00200000.pkl --gen_cd /scratch/ag4508/unit/exps/smiling_bigK/smiling_bigK_gen_00200000.pkl --dis_cd /scratch/ag4508/unit/exps/smiling_bigK/smiling_bigK_dis_00200000.pkl 

#python cocogan_train_fourway.py --config ../exps/unit/anant_smiling_haircolorbigK_4wayjoint.yaml --log /scratch/ag4508/unit/log/ 
#python cocogan_train.py --config ../exps/unit/anant_blondbrunette_nosmiling.yaml --log /scratch/ag4508/unit/log/ 

#python cocogan_train_fourway.py --config ../exps/unit/anant_smiling_blondbrunette_bigK_warmstart.yaml --log /scratch/ag4508/unit/log/ --warm_start 1 --gen_ab /scratch/ag4508/unit/exps/blondbrunette_smiling_bigK/blondbrunette_smiling_bigK_gen_00200000.pkl --dis_ab  /scratch/ag4508/unit/exps/blondbrunette_smiling_bigK/blondbrunette_smiling_bigK_dis_00200000.pkl --gen_cd /scratch/ag4508/unit/exps/smiling_bigK/smiling_bigK_gen_00200000.pkl --dis_cd /scratch/ag4508/unit/exps/smiling_bigK/smiling_bigK_dis_00200000.pkl 
