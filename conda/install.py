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
    ]

    # Execute conda commands
    for command in commands:
        install_command(command)
