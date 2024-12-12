#!/bin/bash

#SBATCH -J "clustering_agent_latent_space_name"
#SBATCH --time 01:00:00
#SBATCH --mem-per-gpu 50000

#SBATCH --partition hpda_mig
#SBATCH --gres gpu:a100_1g.10gb

#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=8
#SBATCH --output /home/2024016/malaar01/Programming_Projects/Clustering_Agent_Latent_Space/criann_logs/%J_%x.out
#SBATCH --error  /home/2024016/malaar01/Programming_Projects/Clustering_Agent_Latent_Space/criann_logs/%J_%x.err

PROJECTNAME=Clustering_Agent_Latent_Space
module purge
module load aidl/pytorch/1.11.0-cuda11.3
export PYTHONUSERBASE=~/packages/$PROJECTNAME
export PATH=$PATH:~/packages/$PROJECTNAME

# rsync -av --exclude='./temporary' --exclude='./experiments' --exclude='criann_logs' . $LOCAL_WORK_DIR
echo Local Working directory : $LOCAL_WORK_DIR

cp -R . $LOCAL_WORK_DIR
cd $LOCAL_WORK_DIR || exit
cd ./Clustering_Agent_Latent_Space || exit

echo Working directory : $PWD

params=(
  --configuration_file_path ./configurations/experimentation/
)

python3 ./executables/reinforcement_learning_training.py ${params[@]}
