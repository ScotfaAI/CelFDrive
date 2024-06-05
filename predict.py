from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import contextlib
import os
import math
import argparse
from datetime import datetime
today_date = datetime.now().strftime("%Y-%m-%d")

repo_path = "path\\to\\CelFDrive"

model = YOLO(repo_path + "\\models\\SiRDNA\\20240514_Polled_0.0_4\\weights\\best.pt") # absolute path

today_date = datetime.now().strftime("%Y-%m-%d")
logging_directory = repo_path+f"\\Logging\\{today_date}" # absolute path
experiment_path = None



# Function to find the next experiment folder name and create it
def create_exp_folder(base_dir):
    global experiment_path
    # Get all items in the base directory
    items = os.listdir(base_dir)
    
    # Filter out items that are not directories or don't match the 'exp###' pattern
    exp_folders = [item for item in items if item.startswith('exp') and item[3:].isdigit() and os.path.isdir(os.path.join(base_dir, item))]
    
    # Sort the folders to find the highest number
    exp_folders.sort()
    
    # Determine the next experiment number
    if exp_folders:
        # Extract the highest current number and add 1
        last_exp_num = int(exp_folders[-1][3:])
        next_exp_num = last_exp_num + 1
    else:
        # If no such folder exists, start with 1
        next_exp_num = 1
    
    # Format the new folder name
    new_folder_name = f"exp{next_exp_num:03d}"
    
    # Create the new experiment folder
    experiment_path = os.path.join(base_dir, new_folder_name)
    os.makedirs(experiment_path, exist_ok=True)
    
    
# Function to find the next experiment folder name and create it
def get_outimg_path():
    global experiment_path
    # Get all items in the base directory
    # print(experiment_path)
    items = os.listdir(experiment_path)
    # print(items)
    
    # Filter out items that are not directories or don't match the 'tmpimg###' pattern
    tmp_images = [item for item in items if item.startswith('tmpimg') and item[-7:-4].isdigit() and os.path.isfile(os.path.join(experiment_path, item))]
    # print(tmp_images)
    # Sort the folders to find the highest number
    tmp_images.sort()
    
    # Determine the next experiment number
    if tmp_images:
        # Extract the highest current number and add 1
        last_img_num = int(tmp_images[-1][-7:-4])
        next_img_num = last_img_num + 1
    else:
        # If no such folder exists, start with 1
        next_img_num = 1
    
    # Format the new folder name
    new_file_name = f"tmpimg{next_img_num:03d}.png"
    
    # Create the new experiment folder
    img_path = os.path.join(experiment_path, new_file_name)
    return img_path


def filter_and_sort_detections(detections, class_info):
    """
    Filters and sorts object detection results based on class-specific confidence
    levels and importance rankings.

    Parameters:
    - detections: List of detection results in the format [cls, x, y, w, h, confidence].
    - class_info: Dictionary with class id as keys and tuples as values, where each tuple
                  contains (class name, acceptable confidence level, ranking).

    Returns:
    - A list of filtered and sorted detections.
    """
    # Filter detections by acceptable confidence level
    filtered_detections = [
        det for det in detections
        if det[5] >= class_info[det[0]][1] and class_info[det[0]][2] != -1
    ]

    # Sort detections by ranking and then by confidence (descending)
    sorted_detections = sorted(
        filtered_detections,
        key=lambda det: (class_info[det[0]][2], -det[5])
    )

    return sorted_detections

def global_filter_and_sort_detections(all_detections, class_info):
    """
    Filters and sorts all detections across the entire dataset.

    Parameters:
    - all_detections: List of all detections with their details.
    - class_info: Dictionary with class-specific configuration.

    Returns:
    - List of globally filtered and sorted detections.
    """
    # Apply global filter based on class-specific confidence thresholds
    filtered_detections = [
        det for det in all_detections if det[3] >= class_info[det[4]][1]
    ]

    # Sort globally by class ranking and confidence
    sorted_detections = sorted(
        filtered_detections, key=lambda det: (class_info[det[4]][2], -det[3])
    )
    return sorted_detections


def plot_image_with_results(image, boxes, class_names, class_info, file_path):
    """
    Plots an image with bounding boxes and class:confidence labels.

    Parameters:
    - image: 2D array representing the image.
    - boxes: List of lists, where each inner list contains:
             [class, x, y, w, h, confidence]
             Coordinates and sizes are not normalized.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')  # Assuming the image is in grayscale
    ax.axis('off')  # Turn off the axis

    filtered_boxes = filter_and_sort_detections(boxes, class_info)


    for box in filtered_boxes:
        class_id, x, y, w, h, confidence = box
        # Create a Rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')

        # Add the rectangle to the plot
        ax.add_patch(rect)

        # Add label
        label = f"{class_names[class_id]}:{confidence:.2f}"
        ax.text(x, y, label, color='white', fontsize=8, ha='left', va='bottom',
                bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.5))

    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to prevent it from displaying in the notebook

    

def preprocess_image(img):
    """Handles greyscale or RBG then flattens the top x % of the image. 
    Then normalise based on min and max.

    Args:
        img (np.array): Image to preprocess.

    Returns:
        np.array: Processed image.
    """

    percentile = 0.01
    percentile_threshold = 100 - percentile  # 100 - 0.3
    threshold_value = np.percentile(img, percentile_threshold)
    # we assume 2D here so if its 3D at this point the 3rd index we assume is channels that are identical as others for saving
    if len(img.shape) == 3:
        h, w, _ = img.shape
        timepoint_data = img[:,:, 0]
    elif len(img.shape) == 2:
        h, w = img.shape
        timepoint_data = img
    else:
        print("Shape Error in predict.preprocess_image")


    timepoint_data_normalized = np.where(timepoint_data > threshold_value, threshold_value, timepoint_data)

    # Normalize the image to the range of 0 to 1

    timepoint_data_normalized2 = (timepoint_data_normalized - timepoint_data_normalized.min()) / (
        timepoint_data_normalized.max() - timepoint_data_normalized.min())
    # plt.imshow(timepoint_data_normalized2)

    img = (timepoint_data_normalized2 * 255).astype(np.uint8)

    return img

def split_image(img):
    """
    Splits the image into 640x640 images
    """
    # this should be 2D by this point
    height, width = img.shape

    desired_im_size = 640
    split_images = []
    for row in range(math.ceil(height/desired_im_size)):  # Two rows
        for col in range(math.ceil(width/desired_im_size)):  # Three columns
            # initial topLeft
            x1 = col * desired_im_size
            y1 = row * desired_im_size
            # calculate and limit bottomRight
            x2 = min(x1+desired_im_size, width)
            y2 = min(y1+desired_im_size, height)
            # recalculate topLeft
            x1 = x2 - desired_im_size
            y1 = y2 - desired_im_size
            # get sub image
            split_img = img[y1:y2, x1:x2]
            split_images.append((split_img, x1, y1))

    return split_images

def adjust_coordinates(detections, x_offset, y_offset):
    """
    Adjusts the detection coordinates to the original image's coordinate system.
    """
    detections_adjusted=[]
    for detection in detections:
        box =[]
        #print(detection.xyxy)
        box.append(int(detection.cls))
        box.append(float(detection.xyxy[:, 0] + x_offset))  # Adjust x coordinate
        box.append(float(detection.xyxy[:, 1] + y_offset))  # Adjust y coordinate
        box.append(float(detection.xywh[:, 2]))  # Adjust x coordinate
        box.append(float(detection.xywh[:, 3]))  # Adjust y coordinate
        box.append(float(detection.conf))
        # print(box)
        detections_adjusted.append(box)
    return detections_adjusted

def process_image(raw_img, conf, save_path, class_info, plot = False):
    processed_img = preprocess_image(raw_img)
    split_images = split_image(processed_img)
    results = []

    for img, x_offset, y_offset in split_images:

        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        #Suppress output to prevent matlab crashing slidebook
        with open(os.devnull, 'w') as nullfile:
             with contextlib.redirect_stdout(nullfile):

                results_split = model(img, conf = conf)

            # Adjust coordinates
        if len(results_split[0].boxes.xyxy)!= 0:
            results_adjusted = adjust_coordinates(results_split[0].boxes, x_offset, y_offset)
            results.extend(results_adjusted)

    if plot:
        class_names = {key: value[0] for key, value in class_info.items()}
        plot_image_with_results(processed_img, results, class_names, class_info, save_path)

    return results

def process_image_from_path(image_path, conf, save_path, plot = False):
    img = cv2.imread(image_path)
    results = process_image(img, conf, save_path, plot)

    return results



def process_single_location(x,y,z, image, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, z_offset, class_info):
    """
    Process a single 3D location to convert image coordinates to physical coordinates.

    Parameters:
    x (float): The x-coordinate in physical space.
    y (float): The y-coordinate in physical space.
    z (float): The z-coordinate in physical space.
    image (array): The 3D or 2D image data.
    xy_pixel_spacing (float): The pixel spacing in the x and y directions.
    z_spacing (float): The spacing in the z direction.
    x_stage_direction (int): The direction of the stage movement in the x direction.
    y_stage_direction (int): The direction of the stage movement in the y direction.
    z_stage_direction (int): The direction of the stage movement in the z direction.
    LLSM (bool): Flag indicating if Lattice Light-Sheet Microscopy (LLSM) adjustments are needed.
    z_offset (float): The z-coordinate offset.
    class_info (dict): A dictionary containing class information.

    Returns:
    list: A list of converted results containing adjusted coordinates and additional metadata.
    """
    new_z = z+z_offset
    if len(image.shape)==3:
        image = np.max(image, z)
    
    # return 2d image
    height, width = image.shape
    # this will cause issues with relative paths
    img_path = get_outimg_path()
    
    class_names = {key: value[0] for key, value in class_info.items()}

    results = process_image(image, 0.01, img_path, class_info , plot = True)

    # Filter and sort the detections
    filtered_sorted_detections = filter_and_sort_detections(results, class_info)

    converted_results = []
    for result in filtered_sorted_detections:
        class_id,im_x,im_y,w,h,conf = result
        # class_id = result[0]
        # im_x, im_y, w, h, conf, class_id = result
        class_name = class_names[class_id]
        converted_results.append(image_cordinates_to_physical(x, y, im_x, im_y, width, height, new_z, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, class_id, conf, class_name))
    
    return converted_results

def image_cordinates_to_physical(x, y, im_x, im_y, w, h, new_z, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, class_id, conf, class_name):
    """
    Convert image coordinates to physical coordinates.

    Parameters:
    x (float): The x-coordinate in physical space.
    y (float): The y-coordinate in physical space.
    im_x (float): The x-coordinate in image space.
    im_y (float): The y-coordinate in image space.
    w (float): The width of the image.
    h (float): The height of the image.
    new_z (float): The z-coordinate in physical space.
    xy_pixel_spacing (float): The pixel spacing in the x and y directions.
    z_spacing (float): The spacing in the z direction.
    x_stage_direction (int): The direction of the stage movement in the x direction.
    y_stage_direction (int): The direction of the stage movement in the y direction.
    z_stage_direction (int): The direction of the stage movement in the z direction.
    LLSM (bool): Flag indicating if Lattice Light-Sheet Microscopy (LLSM) adjustments are needed.
    class_id (int): The class ID.
    conf (float): The confidence score.
    class_name (str): The class name.

    Returns:
    list: A list containing the adjusted x, y, and z coordinates, confidence score, class ID, and class name.
    """

    # adjust for x and y being centred on the image by adding an offset of half the image to the location
    x_offset = im_x - w/2;
    y_offset = im_y - h/2;
    
    # Y stage direction is not correct in LLSM.
    if LLSM:
        adjusted_x = x + x_offset * xy_pixel_spacing * x_stage_direction;
        adjusted_y = y + y_offset * xy_pixel_spacing * y_stage_direction*(-1);
    else:
        adjusted_x = x + x_offset * xy_pixel_spacing * x_stage_direction;
        adjusted_y = y + y_offset * xy_pixel_spacing * y_stage_direction;
    
    return [adjusted_x, adjusted_y, new_z, conf, class_id, class_name]

def merge_close_coordinates(coordinates, tolerance):
    """
    Merge coordinates that are within a specified tolerance and retain the class name from the first instance.

    Parameters:
    coordinates (array): Array of coordinates and class names.
    tolerance (float): The radius within which coordinates should be considered the same.

    Returns:
    tuple: Four separate lists for x, y, z coordinates, and class names of the unique locations.
    """
    unique_coords = []

    for coord in coordinates:
        x, y, z, conf, class_id, class_name = coord
        if not any(np.sqrt((x - uc[0])**2 + (y - uc[1])**2) <= tolerance for uc in unique_coords):
            unique_coords.append(coord)

    unique_coords = np.array(unique_coords, dtype=object)
    return unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2], list(unique_coords[:, 5])

    


def process_montage(X,Y,Z, image, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, z_offset, class_info):
    """
    Process a montage of images to extract and merge close physical coordinates and their class names.

    Parameters:
    X, Y, Z (list of floats): Lists of x, y, z coordinates.
    image (array): The complete montage image data.
    xy_pixel_spacing, z_spacing (float): Pixel dimensions in physical units.
    x_stage_direction, y_stage_direction, z_stage_direction (int): Stage direction multipliers.
    LLSM (bool): Flag for Lattice Light Sheet Microscopy specific adjustments.
    z_offset (float): Offset to apply to z coordinates.
    micron_tolerance (float): Distance tolerance in microns for merging coordinates.
    class_names (list of str): List of class names corresponding to each coordinate.

    Returns:
    tuple: Four arrays containing the new x, y, z coordinates, and their respective class names.
    """
    
    results = []
    image_array = np.array(image)
    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        tmp = process_single_location(x, y, z, image_array[:,:,i], xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, z_offset, class_info)
        if tmp:
            for item in tmp:
                results.append(item) 
    
    sorted_results = global_filter_and_sort_detections(results, class_info)
    
    if len(results) == 0:
        return np.array([]), np.array([]), np.array([]), []  # no results to process

    final_result = np.array(sorted_results, dtype=object)

    new_X, new_Y, new_Z, new_class_names = merge_close_coordinates(final_result, 20)
    return new_X, new_Y, new_Z, new_class_names



def get_target_location(X,Y,Z, image, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM = False, z_offset = 0):
    global experiment_path

    """
    Get the target location for image capture based on the given coordinates and settings.

    Parameters:
    X (list of float): List of x-coordinates in physical space.
    Y (list of float): List of y-coordinates in physical space.
    Z (list of float): List of z-coordinates in physical space.
    image (array): The image data.
    xy_pixel_spacing (float): The pixel spacing in the x and y directions.
    z_spacing (float): The spacing in the z direction.
    x_stage_direction (int): The direction of the stage movement in the x direction.
    y_stage_direction (int): The direction of the stage movement in the y direction.
    z_stage_direction (int): The direction of the stage movement in the z direction.
    LLSM (bool): Flag indicating if Lattice Light-Sheet Microscopy (LLSM) adjustments are needed.
    z_offset (float): The z-coordinate offset.
    class_info (dict, optional): A dictionary containing class information. Defaults to predefined classes if None.

    Returns:
    tuple: A tuple containing:
        - N (int): The number of locations.
        - new_X (list of float): List of new x-coordinates.
        - new_Y (list of float): List of new y-coordinates.
        - new_Z (list of float): List of new z-coordinates.
        - script_list (list): List of scripts for capturing images.
        - name_list (list): List of names for the next capture, indicating the class found or 'nothing found'.
        - comment_list (list): List of comments.
    """

    # convert to a list if just 1 location
    if type(X) == float:
        X = [X]
        Y = [Y]
        Z = [Z]

    
    if LLSM:
        class_info = {
        0: ('prophase', 0.1, 1),
        1: ('earlyprometaphase', 0.2, 0),
        2: ('prometaphase', 1.0, 2),
        3: ('metapase', 1.0, 3),
        4: ('anaphse', 1.0, 4),
        5: ('telophase', 1.0, 5)
        }
    else:
        class_info = {
        0: ('prophase', 0.01, 1),
        1: ('earlyprometaphase', 0.01, 0),
        2: ('prometaphase', 0.01, 2),
        3: ('metapase', 0.01, 3),
        4: ('anaphse', 0.01, 4),
        5: ('telophase', 0.05, 5)
        }


    # Ensure the logging directory exists (simulate the given directory structure)
    os.makedirs(logging_directory, exist_ok=True)
    # Create the next experiment folder and get its path
    create_exp_folder(logging_directory)


    new_X,new_Y,new_Z, class_list = process_montage(X,Y,Z, image, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, z_offset, class_info)

    N = len(new_X)

    if N == 0:
        N = 1
        script_list = ["donothing"]
        new_X = np.array([X[0]])
        new_Y = np.array([Y[0]])
        new_Z = np.array([Z[0]])
        name_list = [f"nothing"]
        comment_list = ["nothing"]
    else:
        if LLSM:
            script_list = ["find-loc-out" for i in range(N)]
        else:
            script_list = ["floifmHighres" for i in range(N)]
        name_list = [f"{class_list[i]} Highres x {new_X[i]}, Y {new_Y[i]}, Z {new_Z[i]} " for i in range(N)]
        comment_list = ["Highres" for i in range(N)]
    
    return N, new_X, new_Y, new_Z, script_list, name_list, comment_list


