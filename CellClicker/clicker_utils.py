import re
import os

def convert_path_format(input_path):
    if os.name == 'posix':
        # Running on macOS or Linux, convert to macOS-style path
        return input_path.replace('\\', '/')
    else:
        # Running on Windows, convert to Windows-style path
        return input_path.replace('/', '\\')

def get_previous_image_name(image_name):
    """Decrease the timepoint of the image by one and return the new image name."""
    # Extract the timepoint using regular expression
    match = re.search(r't(\d{3})\.png$', image_name)
    if match:
        timepoint = int(match.group(1))
        # If timepoint is already 0, return  None
        if timepoint == 0:
            return None
        
        timepoint -=1
        # Replace the old timepoint with the new one in the image name
        new_image_name = re.sub(r't\d{3}\.png$', f't{timepoint:03}.png', image_name)
        return new_image_name
    else:
        return None  # Return None if the format doesn't match
    
def get_relative_image_name(image_name, stepback):
    """Decrease the timepoint of the image by stepback and return the new image name."""
    # Extract the timepoint using regular expression
    match = re.search(r't(\d{3})\.png$', image_name)
    if match:
        timepoint = int(match.group(1))
        # If adjusted timepoint is less than 0 return None
        timepoint -= stepback
        if timepoint < 0:
            return None
        # Replace the old timepoint with the new one in the image name
        new_image_name = re.sub(r't\d{3}\.png$', f't{timepoint:03}.png', image_name)
        return new_image_name
    else:
        return None  # Return None if the format doesn't match
    
def get_relative_label_name(image_name, stepback):
    """Decrease the timepoint of the image by stepback and return the new image name."""
    # Extract the timepoint using regular expression
    match = re.search(r't(\d{3})\.txt$', image_name)
    if match:
        timepoint = int(match.group(1))
        # If timepoint is already 0, keep it as is (or handle accordingly)
        timepoint = max(0, timepoint - stepback)
        # Replace the old timepoint with the new one in the image name
        new_image_name = re.sub(r't\d{3}\.txt$', f't{timepoint:03}.txt', image_name)
        return new_image_name
    else:
        return None  # Return None if the format doesn't match
    
    
def append_yolov5_label(label_path, x_center, y_center, width, height, img_width, img_height, class_id):
    """Append a new YOLOv5 label to a label file, removing any existing newline characters."""
    # Normalize the coordinates
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    # Read the existing content of the file
    with open(label_path, 'r') as f:
        existing_content = f.read()

    # Remove newline characters if they exist
    existing_content = existing_content.replace("    \n", "")

    # Append the new label and add a newline
    new_label = f"{class_id} {x_center} {y_center} {width} {height}\n"

    # Append the new label to the file
    with open(label_path, 'a') as f:
        f.write(new_label)