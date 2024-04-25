import tkinter as tk
import os
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from .name_selector import run_name_selector
from .manageXML import append_cell_regions_xml, find_labels_and_extract_rois, get_all_label_names, get_series_count_for_label, get_all_images, cell_xml_to_dataframe
from .user_xml import store_results, store_results_multiclass, read_xml_to_dataframe
from .convert_selections import modify_class_ids, append_modified_labels
import numpy as np

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

def load_selector(image_dict, set_index):
    selected_indices = []
    image_sets = []

    for (image_name, series_id) in image_dict.keys():
        image_sets.append(image_dict[(image_name, series_id)])

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    print(f"Loading {len(image_sets)} image sets")
    display_set(image_sets, set_index, selected_indices, root, "prometaphase")
    root.mainloop()
    return selected_indices

# def normalize_image(image):
#     image = (image - image.min()) / (image.max() - image.min()) * 255
#     return image.astype(np.uint8)

def normalize_image(image):
        """Applies CLAHE to an image to enhance contrast locally."""
        # Convert image to grayscale if it is in color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create a CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(image)
    
        return cl1

def display_set(image_sets, set_index, selected_indices, root, phase):
    window = tk.Toplevel()
    window.title(f"Select Image for {phase.capitalize()} from Set {set_index + 1}")
    series = image_sets[set_index]
    set_len = len(series)

    photo_images = []  # To store PhotoImage references and prevent garbage collection

    for i, img_array in enumerate(series):
        img_array = normalize_image(img_array)
        img = Image.fromarray(img_array)
        img.thumbnail((100, 100))
        img_tk = ImageTk.PhotoImage(img)
        photo_images.append(img_tk)

        btn = tk.Button(window, image=img_tk, command=lambda i=i: on_selection_clicked(i, window, image_sets, selected_indices, root, set_len, phase, set_index))
        btn.image = img_tk  # Keep a reference
        btn.grid(row=0, column=i)

    window.geometry("+100+100")  # Optional: Position the window at a specific location

    # Buttons for skipping phase and marking blurry
    skip_btn = tk.Button(window, text="Skip Phase", command=lambda: on_skip_clicked(window, image_sets, selected_indices, root, phase, set_index))
    skip_btn.grid(row=1, column=0, sticky='ew')

    blurry_btn = tk.Button(window, text="Mark as Blurry", command=lambda: on_blurry_clicked(window, image_sets, selected_indices, root, set_index))
    blurry_btn.grid(row=1, column=1, sticky='ew', columnspan=set_len)

def on_selection_clicked(index, window, image_sets, selected_indices, root, set_len, phase, set_index):
    if len(selected_indices) <= set_index:
        selected_indices.append({})

    selected_indices[set_index][phase] = index
    print(f"Selected {phase} in set {set_index + 1}: {index}")
    window.destroy()
    handle_next_phase_or_set(image_sets, selected_indices, root, phase, set_index)

def handle_next_phase_or_set(image_sets, selected_indices, root, phase, set_index):
    phases = ['prometaphase', 'metaphase', 'anaphase']
    next_index = phases.index(phase) + 1
    if next_index < len(phases):
        display_set(image_sets, set_index, selected_indices, root, phases[next_index])
    else:
        if set_index + 1 < len(image_sets):
            display_set(image_sets, set_index + 1, selected_indices, root, phases[0])
        else:
            messagebox.showinfo("Completed", "All selections completed.")
            print("Final selections:", selected_indices)
            root.quit()

def on_blurry_clicked(window, image_sets, selected_indices, root, set_index):
    if len(selected_indices) <= set_index:
        selected_indices.append({phase: 'blurry' for phase in ['prometaphase', 'metaphase', 'anaphase']})

    window.destroy()
    if set_index + 1 < len(image_sets):
        display_set(image_sets, set_index + 1, selected_indices, root, "prometaphase")
    else:
        messagebox.showinfo("Completed", "All selections completed.")
        root.quit()

def on_skip_clicked(window, image_sets, selected_indices, root, phase, set_index):
    if len(selected_indices) <= set_index:
        selected_indices.append({})

    selected_indices[set_index][phase] = 'skipped'
    print(f"Skipped {phase} in set {set_index + 1}")
    window.destroy()
    handle_next_phase_or_set(image_sets, selected_indices, root, phase, set_index)



        


def load_ui(cell_xml):
    # Call the UI function to run the name selector
    name_xml = run_name_selector("select_xmls")
    print(f"Selected XML: {name_xml}")
    images_dict = get_all_images(cell_xml)
    selected_indicies = load_selector(images_dict, 0)
    store_results(images_dict, selected_indicies, name_xml)

def load_ui_from_folder():
    # Call the UI function to run the name selector

    directory = filedialog.askdirectory(title="Select Directory with Images")
    if not directory:
        return
    selections_folder = os.path.join(directory, "user_selections")
    os.makedirs(selections_folder, exist_ok=True)
    
    cell_xml = os.path.join(os.path.join(directory, "images"), "cell_reigons.xml")
    cell_xml = os.path.normpath(cell_xml)

    name_xml = run_name_selector(selections_folder)
    print(f"Selected XML: {name_xml}")
    images_dict = get_all_images(cell_xml)
    selected_indicies = load_selector(images_dict, 0)
    print(selected_indicies)
    store_results_multiclass(images_dict, selected_indicies, name_xml)
    
def xml_to_labels(name_xml, cell_xml):
    
    user_df = read_xml_to_dataframe(name_xml)
    
    cell_df = cell_xml_to_dataframe(cell_xml)
    
    target_class_id = 2
    modified_df = modify_class_ids(cell_df, user_df, target_class_id)
    append_modified_labels(modified_df)
    
    
def debug_xml_to_labels(name_xml, cell_xml):
    
    user_df = read_xml_to_dataframe(name_xml)
    
    cell_df = cell_xml_to_dataframe(cell_xml)
    return user_df, cell_df

def debug_xml_to_labels2(name_xml, cell_xml):
    
    user_df = read_xml_to_dataframe(name_xml)
    
    cell_df = cell_xml_to_dataframe(cell_xml)
    
    target_class_id = 2
    modified_df = modify_class_ids(cell_df, user_df, target_class_id)
    return modified_df
#     target_class_id = 2
#     modified_df = modify_class_ids(cell_df, user_df, target_class_id)
#     append_modified_labels(modified_df)
        
