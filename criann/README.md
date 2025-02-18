# **Connect to CRIANN**
```bash
ssh -l malaar01 austral.criann.fr
cd ~/Programming_Projects/Clustering_Agent_Latent_Space/
```

---
# **Git**

```git diff```: Shows unstaged changes in your working directory compared to the last commit.

```git pull```: Updates your local repository with changes from a remote repository.

```git push```: Uploads your local commits to a remote repository.

```git stash```: Temporarily saves uncommitted changes to allow for switching tasks or branches.

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

4. **Read the logs of a job**  
   ```bash
   less ~/Programming_Projects/Clustering_Agent_Latent_Space/criann_logs/<file_name>
   ```  

5. **Follow the logs of a job in real-time**  
   ```bash
   tail -f ~/Programming_Projects/Clustering_Agent_Latent_Space/criann_logs/<file_name>
   ```  

6. **Connection to the compute node of the job**
   ```bash
   ssh <NODELIST(REASON)>
   ```

7. **Job's history**
   ```bash
   sacct
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

# Setup Environment

```bash
./criann/python_environment/install_packages_criann.sh
```

# Retrieving results
This command uses rsync to copy files from a remote server to a local directory.
```bash
rsync -avz malaar01@austral.criann.fr:~/Programming_Projects/Clustering_Agent_Latent_Space/experiments/ ./experiments/criann/
```
This script runs an rsync command every 5 minutes (300 seconds) to keep the local directory.
```bash
while true; do rsync -avz malaar01@austral.criann.fr:~/Programming_Projects/Clustering_Agent_Latent_Space/experiments/ ./experiments/criann/; sleep 300; done
```