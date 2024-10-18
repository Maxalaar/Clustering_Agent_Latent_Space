import subprocess

# Name of the environment
conda_environment_name = "Clustering_Agent_Latent_Space"

# Create the conda environment
subprocess.run(f"conda create -n {conda_environment_name} -y", shell=True, check=True)

# Activate the environment and set PYTHONPATH
subprocess.run(f"conda activate {conda_environment_name} && conda env config vars set PYTHONPATH='.'", shell=True, check=True)

# Restart the environment for the changes to take effect
subprocess.run(f"conda deactivate && conda activate {conda_environment_name}", shell=True, check=True)

# Confirmation
print(f"The environment {conda_environment_name} has been created, PYTHONPATH has been set, and the environment has been restarted.")