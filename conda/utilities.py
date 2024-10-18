import subprocess


def remove_conda_environment(env_name):
    try:
        subprocess.run(['conda', 'env', 'remove', '--name', env_name], check=True)
        print(f'The existing environment "{env_name}" has been successfully removed.')
    except subprocess.CalledProcessError as error:
        print(f'An error occurred while removing the environment "{env_name}": {str(error)}.')


def get_environment_conda_name(configuration_path):
    env_name = None
    with open(configuration_path, 'r') as file:
        for line in file:
            if line.startswith('name:'):
                env_name = line.split(':')[1].strip()
                break
    return env_name


def install_command(command):
    try:
        print('-- --')
        print(f'Running command: {command}')
        subprocess.run(command, shell=True, check=True)
        print(f'Successfully ran command: {command}')
        print()
    except subprocess.CalledProcessError as error:
        print(f'An error occurred while running command: {command}. Error: {str(error)}')
    except FileNotFoundError as fnf_error:
        print(f'Command not found: {command}. Error: {str(fnf_error)}')


def save_conda_environment(path):
    try:
        subprocess.run(['conda', 'env', 'export', '--no-builds', '--file', path], check=True)
        print('The environment has been successfully saved to ' + path + '.')

        # Remove the prefix line from the exported file
        with open(path, 'r') as file:
            lines = file.readlines()

        with open(path, 'w') as file:
            for line in lines:
                if not line.strip().startswith('prefix:'):
                    file.write(line)

        print('The prefix line has been successfully removed from ' + path + '.')
    except subprocess.CalledProcessError as error:
        print(f'An error occurred while saving the environment: {error}')
    except Exception as error:
        print(f'An unexpected error occurred: {error}')


def restore_conda_environment(configuration_path):
    try:
        subprocess.run(['conda', 'env', 'update', '--file', configuration_path, '--prune'], check=True)
        print(f'The environment has been successfully restored from {configuration_path}.')
    except subprocess.CalledProcessError as error:
        print(f'An error occurred while restoring the environment: {str(error)}.')
    except Exception as error:
        print(f'An unexpected error occurred: {str(error)}')
