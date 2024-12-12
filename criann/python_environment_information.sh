PROJECTNAME=Clustering_Agent_Latent_Space
module purge
module load aidl/pytorch/2.2.0-cuda12.1
export PYTHONUSERBASE=~/packages/$PROJECTNAME
export PATH=$PATH:~/packages/$PROJECTNAME/

python3 ./python_environment/python_environment_information.py