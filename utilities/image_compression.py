from PIL import Image
import numpy as np
from PIL.Image import Resampling


def image_compression(image, new_width=None, new_height=None, grayscale=False):
    # Convert the numpy array to a PIL Image object
    image = Image.fromarray(image)

    # Calculate new dimensions while keeping the aspect ratio
    if new_width and not new_height:
        # Calculate height based on specified width
        aspect_ratio = image.height / image.width
        new_height = int(new_width * aspect_ratio)
    elif new_height and not new_width:
        # Calculate width based on specified height
        aspect_ratio = image.width / image.height
        new_width = int(new_height * aspect_ratio)
    elif new_width and new_height:
        # Adjust dimensions to keep the original aspect ratio
        width_ratio = new_width / image.width
        height_ratio = new_height / image.height
        scale_factor = min(width_ratio, height_ratio)
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)

    # Resize the image if new dimensions are defined
    if new_width and new_height:
        image = image.resize((new_width, new_height), Resampling.BILINEAR)

    # Convert to grayscale if requested
    if grayscale:
        image = image.convert("L")

    # Return as a numpy array
    return np.array(image)
