# Connect to CRIANN
Creat ssh key and add it on the CRIANN:
```bash
ssh-keygen -t rsa -b 4096 -C "maxalaar@protonmail.com"
ssh-copy-id -i ~/.ssh/id_rsa.pub malaar01@austral.criann.fr
```


```bash
ssh -l malaar01 austral.criann.fr
cd ./Programming_Projects/Clustering_Agent_Latent_Space/
git pull
git reset --hard HEAD
```

# Setup CRIANN Environment

To set up the required environment for this project, follow these steps:

1. **Set authorizations:**
```bash
chmod +x ./criann/setup_criann_environment.sh
```

2. **Create a Environment:**
```bash
./criann/setup_criann_environment.sh
```

# Load the Environment
```bash
PROJECTNAME=Clustering_Agent_Latent_Space
module purge
module load aidl/pytorch/2.2.0-cuda12.1
export PYTHONUSERBASE=~/packages/$PROJECTNAME
export PATH=$PATH:~/packages/$PROJECTNAME
```

Test if the environment is correct :
```bash
python3 ./conda/python_environment_information.py
```