import importlib
from pathlib import Path


def get_configuration_class(configuration_file_path: str):
    configuration_file_path = Path(configuration_file_path)

    # Convert the configuration path to the correct module path (replace slashes with dots and remove the .py extension)
    module_path = str(configuration_file_path.with_suffix('')).replace('/', '.')
    print('Configuration load : ' + str(module_path))

    # Import the module dynamically
    module = importlib.import_module(module_path)

    # Assuming the class has the same name as the file (without the .py extension)
    class_name = configuration_file_path.stem
    configuration_class = getattr(module, class_name)

    return configuration_class
