# Conda Environment Setup

To set up the required environment for this project, follow these steps:

1. **Create a Conda Environment:**
```bash
conda create --name Clustering_Agent_Latent_Space python=3.11
```

2. **Activate the Environment**:
```bash
conda activate Clustering_Agent_Latent_Space
```

3. **Set Environment Variables:**

Move to project directory :

```bash
cd .../Clustering_Agent_Latent_Space
```

After activating the environment, set the PYTHONPATH variable to the current directory:
```bash
conda env config vars set PYTHONPATH='.'
conda activate
conda activate Clustering_Agent_Latent_Space
```

4. **Packages:**

Restore packages (best option) :
```bash
cd ./python_environment/
python ./restore.py
```

Or install packages:
```bash
python ./python_environment/install.py
```

# Remove Conda Environment

```bash
conda remove -n Clustering_Agent_Latent_Space --all
```

# GPU

Find the best available version of CUDA on your computer :
```bash
nvidia-smi
```

Find the version of CUDA currently installed
```bash
nvcc --version
```

# Explainability Pipeline

1. Reinforcement Learning Training
```bash
python3 ./executables/reinforcement_learning_training.py \
  --experimentation_configuration_file configurations/experimentation/<name_of_configuration_file.py>
```

2. Video Generation
```bash
python3 ./executables/video_episodes_generation.py \
  --experimentation_configuration_file ./configurations/experimentation/<name_of_configuration_file.py> \
  --reinforcement_learning_path experiments/<path_to_reinforcement_learning_save_directory>
```

3. Generation trajectory dataset
```bash
python3 ./executables/trajectory_dataset_generation.py \
  --experimentation_configuration_file ./configurations/experimentation/<name_of_configuration_file.py> \
  --reinforcement_learning_path experiments/<path_to_reinforcement_learning_save_directory>
```

4. Training surrogate policy
```bash
python3 ./executables/surrogate_policy_training.py \
  --experimentation_configuration_file ./configurations/experimentation/<name_of_configuration_file.py> \
  --trajectory_dataset_path ./experiments/<path_to_trajectory_dataset>
```

Or if you want to continue the training :
```bash
python3 ./executables/surrogate_policy_training.py \
  --experimentation_configuration_file ./configurations/experimentation/<name_of_configuration_file.py> \
  --trajectory_dataset_path ./experiments/<path_to_trajectory_dataset> \
  --surrogate_policy_checkpoint_path experiments/<path_to_surrogate_policy_checkpoint>.ckpt
```

5. Evaluation surrogate policy
```bash
python3 ./executables/surrogate_policy_evaluation.py \
  --experimentation_configuration_file ./configurations/experimentation/<name_of_configuration_file.py> \
  --reinforcement_learning_path experiments/<path_to_reinforcement_learning_save_directory> \
  --surrogate_policy_checkpoint_path experiments/<path_to_surrogate_policy_checkpoint>.ckpt
```

6. Analyse of the latent space
```bash
python3 ./executables/latent_space_analysis.py \
  --experimentation_configuration_file ./configurations/experimentation/<name_of_configuration_file.py> \
  --trajectory_dataset_path ./experiments/<path_to_trajectory_dataset> \
  --surrogate_policy_checkpoint_path experiments/<path_to_surrogate_policy_checkpoint>.ckpt
```