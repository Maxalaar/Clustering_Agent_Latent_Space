from pathlib import Path
import h5py


def display_h5_file_information(h5_file_path: Path):
    try:
        # Open the HDF5 file in read mode
        with h5py.File(h5_file_path, 'r') as h5_file:
            # Display the file name
            print(f"File Information: {h5_file_path}\n")

            # Display groups and datasets
            def inspect_group(group, indent=0):
                for key in group.keys():
                    item = group[key]
                    print(" " * indent + f"{key}:")
                    if isinstance(item, h5py.Group):
                        print(" " * (indent + 2) + "Group")
                        inspect_group(item, indent + 4)
                    elif isinstance(item, h5py.Dataset):
                        print(" " * (indent + 2) + f"Dataset - Shape: {item.shape}, Dtype: {item.dtype}")
                        # Display dataset attributes
                        for attr in item.attrs:
                            print(" " * (indent + 4) + f"Attribute '{attr}': {item.attrs[attr]}")

            inspect_group(h5_file)
            print()
    except Exception as error:
        raise RuntimeError(f"Failed to process the file: {error}")
