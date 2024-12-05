from utilities import restore_conda_environment

if __name__ == '__main__':
    backup_path = './configuration.yml'
    restore_conda_environment(backup_path)

