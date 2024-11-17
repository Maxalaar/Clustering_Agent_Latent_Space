from conda.utilities import install_command

if __name__ == '__main__':
    # List of commands to run
    commands = [
        'conda install -c conda-forge ray-core --yes',
        # 'pip install ray==2.39.0',
        'conda install -c conda-forge ray-default --yes',
        'conda install -c conda-forge ray-data --yes',
        'conda install -c conda-forge ray-train --yes',
        'conda install -c conda-forge ray-tune --yes',
        'conda install -c conda-forge ray-rllib --yes',
        'conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes',
        'conda install conda-forge::tensorflow --yes',
        'conda install -c conda-forge pygame --yes',
        'conda install -c conda-forge gputil --yes ',
        'conda install -c conda-forge tensorboard --yes',
        'conda install conda-forge::moviepy --yes',
        'conda install anaconda::swig --yes',
        'conda install h5py --yes',
        'conda install conda-forge::filelock --yes'
        'conda install conda-forge::pytorch-lightning --yes',
        'conda install conda-forge::gymnasium --yes',
        'conda install -c rapidsai -c conda-forge -c nvidia rapids=24.10 python=3.11 cuda-version=12.1 --yes',
        'conda install bokeh::bokeh --yes',
        'conda install selenium --yes',
        'pip install gymnasium==1.0.0',
        'pip install gymnasium[box2d]',
        'pip install gymnasium[mujoco]',
        'pip install flappy-bird-gymnasium',
        'pip install tetris-gymnasium',
    ]

    # Execute conda commands
    for command in commands:
        install_command(command)
