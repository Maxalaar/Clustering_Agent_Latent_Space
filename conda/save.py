from conda.utilities import save_conda_environment

if __name__ == '__main__':
    backup_path = './configuration.yml'
    save_conda_environment(backup_path)
