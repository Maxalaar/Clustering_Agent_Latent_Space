# Conda Environment Setup

To set up the required environment for this project, follow these steps:

1. **Create a Conda Environment:**
```bash
conda create --name Clustering_Agent_Latent_Space
```

2. **Activate the Environment**:
```bash
conda activate Clustering_Agent_Latent_Space
```

3. **Set Environment Variables:**

After activating the environment, set the PYTHONPATH variable to the current directory:

```bash
conda env config vars set PYTHONPATH='.'
```

4. **Install Packages:**
```bash
python ./conda/restore.py
```

# Remove Conda Environment

```bash
conda remove -n Clustering_Agent_Latent_Space --all
```
