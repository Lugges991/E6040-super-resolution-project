#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./log/c18_l1.out.%j
#SBATCH -e ./log/c18_l1.err.%j
# initial working dir:
#SBATCH -D ./
# Job name:
#SBATCH -J GAN-GPU
# Node feature
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:2
# Number of nodes and MPI tasks per node:
# #SBATCH --cpus-per-task=16
# #SBATCH --ntasks-per-node=1
# wall clock limit(Max. is 24hrs)
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lucas.mahler@tuebingen.mpg.de

module purge 
module load anaconda/3/2020.02
module load cuda/11.2
module load nibabel/2.5.0
# pytorch
module load pytorch/gpu-cuda-11.2/1.8.1

# run
srun python main.py
echo "Jobs finished"

