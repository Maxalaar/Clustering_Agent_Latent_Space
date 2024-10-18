from conda.utilities import install_command

if __name__ == '__main__':
    # List of commands to run
    commands = [
        'conda install -c conda-forge ray-core --yes',
        'conda install -c conda-forge ray-default --yes',
        'conda install -c conda-forge ray-data --yes',
        'conda install -c conda-forge ray-train --yes',
        'conda install -c conda-forge ray-tune --yes',
        'conda install -c conda-forge ray-rllib --yes',
        'conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes',
        'conda install -c conda-forge pygame --yes',
        'conda install -c conda-forge gputil --yes ',
        'conda install -c conda-forge tensorboard --yes',
        'conda install conda-forge::moviepy --yes',
        'conda install anaconda::swig --yes',
        'conda install h5py --yes',
        'conda install conda-forge::pytorch-lightning --yes',
        'conda install conda-forge::gymnasium[all] --yes',
        'pip install gymnasium[box2d]',
        'pip install gymnasium[mujoco]',

        # 'sudo apt install swig',
        # 'pip install gymnasium==1.0.0',
        # 'pip install gymnasium[box2d]',
        # 'pip install gymnasium[mujoco]',
        # 'pip install "numpy<2"',
        # ? -> pip install -U numpy pandas
    ]

    # Execute conda commands
    for command in commands:
        install_command(command)
