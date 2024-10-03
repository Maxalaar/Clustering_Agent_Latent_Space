import ray
import sys
import os
import torch


def environment_information():
    conda_env = os.getenv('CONDA_DEFAULT_ENV')
    if conda_env:
        print('Conda environment name:', conda_env)
    else:
        print('You are not in a Conda environment.')
    print()

    print('Python version:', sys.version)
    print('Ray version:', ray.__version__)
    print('PyTorch version:', torch.__version__)
    print()

    cuda_available = torch.cuda.is_available()
    print(f'Is CUDA available? {cuda_available}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device in use: {device}')
    if cuda_available:
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        print(f'CUDA version: {cuda_version}')
        print(f'cuDNN version: {cudnn_version}')
        print()

        gpu_count = torch.cuda.device_count()
        print(f'Number of available GPUs: {gpu_count}')

        current_gpu = torch.cuda.current_device()
        print(f'Current GPU ID: {current_gpu}')

        gpu_name = torch.cuda.get_device_name(current_gpu)
        print(f'GPU Name: {gpu_name}')
    print()

    ray_initialization = False
    if not ray.is_initialized():
        ray.init()
        ray_initialization = True

    if ray.is_initialized():
        resources = ray.cluster_resources()
        print('Number of CPUs accessible to Ray:', resources.get('CPU', 0))
        print('Number of GPUs accessible to Ray:', resources.get('GPU', 0))
    else:
        print('Ray is not initialized')

    print()
    if ray_initialization:
        ray.shutdown()


if __name__ == '__main__':
    environment_information()
