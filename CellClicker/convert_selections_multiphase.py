import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from .clicker_utils import get_relative_image_name


def xyxy_to_yolov5(x_min, y_min, x_max, y_max, img_width, img_height):
    # Calculate bounding box center
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    
    # Calculate bounding box width and height
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    # Normalize coordinates
    center_x_normalized = center_x / img_width
    center_y_normalized = center_y / img_height
    box_width_normalized = box_width / img_width
    box_height_normalized = box_height / img_height
    
    return center_x_normalized, center_y_normalized, box_width_normalized, box_height_normalized


def adjust_bbox_via_threshold(image_path, label, x_center_norm, y_center_norm, width_norm, height_norm):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at the specified path: {image_path}")

    img_h, img_w = image.shape[:2]
    print(f"Image dimensions: height={img_h}, width={img_w}")

    # Convert YOLO format to absolute bounding box coordinates
    x_center = x_center_norm * img_w
    y_center = y_center_norm * img_h
    width = width_norm * img_w
    height = height_norm * img_h
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    print(f"Initial Bounding Box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

    # Ensure the crop remains within the image boundaries
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, img_w)
    y_max = min(y_max, img_h)

    print(f"Corrected Bounding Box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

    # Crop the relevant part of the image
    crop_img = image[y_min:y_max, x_min:x_max]

    # Convert to grayscale
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return label, x_center_norm, y_center_norm, width_norm, height_norm

    # Find the largest and possibly the second largest contour based on area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    if len(contours) > 1 and cv2.contourArea(contours[1]) >= 0.5 * cv2.contourArea(largest_contour):
        x, y, w, h = cv2.boundingRect(np.vstack([largest_contour, contours[1]]))
    else:
        x, y, w, h = cv2.boundingRect(largest_contour)

    print(f"Contour Bounding Box: x={x}, y={y}, w={w}, h={h}")


    # Adjust coordinates based on the original crop
    x_min_new = max(x_min + x, 0)
    y_min_new = max(y_min + y, 0)
    x_max_new = x_min_new + w
    y_max_new = y_min_new + h

    print(f"New Bounding Box Coordinates: x_min_new={x_min_new}, y_min_new={y_min_new}, x_max_new={x_max_new}, y_max_new={y_max_new}")

    x_center_new, y_center_new, width_new, height_new = xyxy_to_yolov5(x_min_new, y_min_new, x_max_new, y_max_new, img_w, img_h)
    print(w, h)

    print(f"Normalized Output: x_center_new={x_center_new}, y_center_new={y_center_new}, width_new={width_new}, height_new={height_new}")

    return label, x_center_new, y_center_new, width_new, height_new



def parse_xml_for_phases(xml_file):
    """ Parse XML and get phase indices for each image and series. """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_data = {}
    for entry in root.findall('DataEntry'):
        path = entry.find('PathName').text
        series_id = entry.find('SeriesID').text
        indices = {
            phase.tag: int(phase.text) for phase in entry
            if phase.tag in ['prometaphase', 'metaphase', 'anaphase'] and phase.text.isdigit()
        }
        image_data[(path, series_id)] = indices
    return image_data

def parse_xml_for_labels(xml_file):
    """ Parse annotation XML and return structured label data. """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels_data = {}
    for path_element in root.findall('path'):
        image_path = path_element.find('name').text
        for series in path_element.findall('series'):
            series_id = series.get('id')
            labels = []
            for label in series.findall('label'):
                label_info = {
                    'class_id': int(label.find('class_id').text),
                    'x_center': float(label.find('x_center').text),
                    'y_center': float(label.find('y_center').text),
                    'width': float(label.find('width').text),
                    'height': float(label.find('height').text)
                }
                labels.append(label_info)
            labels_data[(image_path, series_id)] = labels
    return labels_data

def generate_filename(base_path, offset):
    """ Generate a new filename based on an offset from the base filename """
    base_name = os.path.basename(base_path)
    prefix, number, ext = base_name.rsplit('_', 2)
    new_number = int(number[1:]) + offset
    new_filename = f"{prefix}_t{str(new_number).zfill(3)}.{ext}"
    return new_filename

def get_phase(new_index, phases):
    # Iterate over the phases
    for phase, value in phases.items():
        # Check if the new index is larger than the corresponding value
        if new_index >= value:
            return phase
    # If the new index is less than or equal to all values, return None
    return None

def create_yolo_labels(phase_data, labels_data, output_dir, user):
    """ Generate YOLO label files considering class_id as filename reference. """
    os.makedirs(output_dir, exist_ok=True)

    class_dict = {'prophase': 0, 'earlyprometaphase':1, 'prometaphase': 2, 'metaphase': 3, 'anaphase': 4 }
    # this gets from cell_reigons the path and series id, and all the indicies
    for (path, series_id), indices in labels_data.items():
        # print(path)
        phases = phase_data.get((path, series_id), [])

# Create a  copy of the dictionary to avoid modifying it while iterating
        phases_copy = phases.copy()

        # Iterate over the dictionary and remove entries where the value is -1
        for key, value in phases_copy.items():
            if value == -1:
                del phases[key]


        print(phases)
        if phases:
            # if prometaphase exists add the early prometaphase class and shift prometaphase along 1
            if 'prometaphase' in phases:
                phases['earlyprometaphase'] = phases['prometaphase']
                phases['prometaphase'] = phases['prometaphase']+1

                # if there is anything before prometphase add the prophase class
                if phases['prometaphase'] > 0:
                    phases['prophase'] = 0


            
            sorted_phases = dict(sorted(phases.items(), key=lambda item: item[1], reverse=True))
            print(sorted_phases)
            
            for label in indices: 

                new_index = len(indices) - int(label['class_id'])
                print(f"newindex {new_index}")
                print(f"looping {label}")
                print(phases)
                
                print(sorted_phases)
                selected_phase = get_phase(new_index, sorted_phases)
                
                print(selected_phase)
                print(class_dict)

                new_class_id = class_dict[selected_phase]
                print(selected_phase)
                print(new_class_id)
                print(path)
                
                filename = get_relative_image_name(path, int(label['class_id']) )
                print(filename)

                adjusted_label = adjust_bbox_via_threshold(filename, new_class_id, label['x_center'], label['y_center'], label['width'], label['height'])
                print(adjusted_label)
                
                # yolo_label = f"{new_class_id} {label['x_center']} {label['y_center']} {label['width']} {label['height']}"
                yolo_label = f"{new_class_id} {adjusted_label[1]} {adjusted_label[2]} {adjusted_label[3]} {adjusted_label[4]}"
                
                # Write label to corresponding file
                filename = filename.replace('.png', '.txt').replace('images', user+'_labels')
                with open(filename, 'a') as file:
                    file.write(yolo_label + '\n')



def convert_selections_multiphase(user_xml, cell_reigons_xml, new_label_folder, user):
# Usage
    phase_data = parse_xml_for_phases(user_xml)
    print(phase_data)
    labels_data = parse_xml_for_labels(cell_reigons_xml)
    print(labels_data)
    create_yolo_labels(phase_data, labels_data, new_label_folder, user)
# phase_data = parse_xml_for_phases('E:/Scott/Data/20240417/user_selections/Sara.xml')
# labels_data = parse_xml_for_labels('E:/Scott/Data/20240417/images/cell_reigons.xml')
# create_yolo_labels(phase_data, labels_data, 'E:/Scott/Data/20240417/new_labels')
# convert_selections_multiphase('E:/Scott/Data/20240417/user_selections/Scott.xml', 'E:/Scott/Data/20240417/images/cell_reigons.xml', 'E:/Scott/Data/20240417/new_labels')