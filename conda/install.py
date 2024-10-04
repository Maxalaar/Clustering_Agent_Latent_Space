from conda.utilities import install_command

if __name__ == '__main__':
    # List of commands to run
    commands = [
        'sudo apt install swig',
        'conda install -c conda-forge ray-core --yes',
        'conda install -c conda-forge ray-default --yes',
        'conda install -c conda-forge ray-data --yes',
        'conda install -c conda-forge ray-train --yes',
        'conda install -c conda-forge ray-tune --yes',
        'conda install -c conda-forge ray-rllib --yes',
        'conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes',
        'conda install -c conda-forge::pygame --yes',
        'pip install gymnasium[box2d]',
        # gputil for GPU
    ]

    # Execute conda commands
    for command in commands:
        install_command(command)
