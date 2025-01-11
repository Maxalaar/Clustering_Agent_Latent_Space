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
   tail -f ~/Programming_Projects/Clustering_Agent_Latent_Space/criann_logs/<file_name>
   ```  

5. **Connection to the compute node of the job**
   ```bash
   ssh <NODELIST(REASON)>
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
```bash
rsync -avz malaar01@austral.criann.fr:~/Programming_Projects/Clustering_Agent_Latent_Space/experiments/ ./experiments/criann/
```

```bash
while true; do rsync -avz malaar01@austral.criann.fr:~/Programming_Projects/Clustering_Agent_Latent_Space/experiments/ ./experiments/criann/; sleep 300; done
```