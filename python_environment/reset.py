import subprocess

from utilities import get_environment_conda_name, remove_conda_environment

if __name__ == '__main__':
    backup_path = './configuration.yml'
    environment_name = get_environment_conda_name(backup_path)
    remove_conda_environment(environment_name)
    subprocess.run(['conda', 'create', '--name', environment_name, 'python=3.11'], check=True)
