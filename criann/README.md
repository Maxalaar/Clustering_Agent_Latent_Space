# **Connect to CRIANN**  
```bash
ssh -l malaar01 austral.criann.fr  
cd ~/Programming_Projects/Clustering_Agent_Latent_Space/  
```

---

# **Submit Jobs**  
1. **Submit a job**  
   ```bash
   sbatch <submission_script>
   ```  

2. **Check your active jobs**  
   ```bash
   squeue --me
   ```  

3. **Navigate to the job's execution directory**  
   ```bash
   cd /dlocal/run/<job_id>
   ```  

4. **Follow the logs of a job in real-time**  
   ```bash
   tail -f ~/Programming_Projects/Clustering_Agent_Latent_Space/criann_log/<file_name>
   ```  

---

# **Cancel Jobs**
1. **Cancel a specific job**  
   ```bash
   scancel <job_id>
   ```

2. **Cancel all your jobs**  
   ```bash
   scancel --me
   ```

---

# **Create and Add an SSH Key**  
1. **Generate an SSH key**  
   ```bash
   ssh-keygen -t rsa -b 4096 -C "maxalaar@protonmail.com"
   ```  

2. **Copy the public key to the CRIANN server**  
   ```bash
   ssh-copy-id -i ~/.ssh/id_rsa.pub malaar01@austral.criann.fr
   ```

[//]: # (ssh c23hpda5)

[//]: # (#git pull)

[//]: # (#git reset --hard HEAD)

[//]: # (# Setup CRIANN Environment)

[//]: # ()
[//]: # (To set up the required environment for this project, follow these steps:)

[//]: # ()
[//]: # (1. **Set authorizations:**)

[//]: # (```bash)

[//]: # (chmod +x ./criann/setup_criann_environment.sh)

[//]: # (```)

[//]: # ()
[//]: # (2. **Create a Environment:**)

[//]: # (```bash)

[//]: # (./criann/setup_criann_environment.sh)

[//]: # (```)

[//]: # ()
[//]: # (# Load the Environment)

[//]: # (```bash)

[//]: # (PROJECTNAME=Clustering_Agent_Latent_Space)

[//]: # (module purge)

[//]: # (module load aidl/pytorch/2.2.0-cuda12.1)

[//]: # (export PYTHONUSERBASE=~/packages/$PROJECTNAME)

[//]: # (export PATH=$PATH:~/packages/$PROJECTNAME)

[//]: # (```)

[//]: # ()
[//]: # (Test if the environment is correct :)

[//]: # (```bash)

[//]: # (python3 ./conda/python_environment_information.py)

[//]: # (```)