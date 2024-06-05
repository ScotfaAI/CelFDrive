import tkinter as tk
import os
import cv2
import numpy as np
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from .name_selector import run_name_selector
from .manageXML import get_all_images, cell_xml_to_dataframe
from .user_xml import store_results, store_results_multiclass, read_xml_to_dataframe
from .convert_selections import modify_class_ids, append_modified_labels
from .convert_selections_multiphase import parse_xml_for_phases_resume

def load_selector(image_dict, set_index, phases, name_xml):
    """Generate the initial UI

    Args:
        image_dict (dict): _description_
        set_index (int): Often set to 0 as we start here.
        phases (list): List of strings for phases
        name_xml (string): Path to Name XML

    Returns:
        list: list of dicts which contain phase selection information
    """

    selected_indices = []
    image_sets = []

    set_count = 0

    for (image_name, series_id) in image_dict.keys():
        set_count+=1
        print(f'{set_count}: {image_name}')
        image_sets.append(image_dict[(image_name, series_id)])

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    print(f"Loading {len(image_sets)} image sets")
    display_set(image_sets, set_index, selected_indices, root, phases[0], phases, name_xml)
    root.mainloop()
    return selected_indices

def normalize_image(image):
    """Normalise image by subtracting the minimum and dividing by the range, 
    then convert to np.uint8.

    Args:
        image (np.uint8): Input image.

    Returns:
        np.uint8 : Normalised image.
    """
    image = (image - image.min()) / (image.max() - image.min()) * 255
    return image.astype(np.uint8)

# def normalize_image(image):
#         """Applies CLAHE to an image to enhance contrast locally."""
#         # Convert image to grayscale if it is in color
#         if len(image.shape) == 3:
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Create a CLAHE object
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         cl1 = clahe.apply(image)
    
#         return cl1

def display_set(image_sets, set_index, selected_indices, root, phase, phases, name_xml):
    """Resumes UI based on what is in name_xml

    Args:
        window (tk.window): Current UI window
        image_sets (list): list of 
        selected_indices (list): list of dicts which contain phase selection information
        root (tk.root): root of TK
        set_index (_type_): _description_
        phases (list): List of strings for phases
        name_xml (string): Path to Name XML
    """

    window = tk.Toplevel()
    window.title(f"Select First frame visible for {phase.capitalize()} - Set {set_index + 1} of {len(image_sets)}")
    
    series = image_sets[set_index]
    set_len = len(series)
    max_images_per_row = 13 
    min_window_width = 100*max_images_per_row  # Calculate minimum width based on max images per row and their thumbnail size
    window.minsize(min_window_width, 0) 

    

    photo_images = []  # To store PhotoImage references and prevent garbage collection

    for i, img_array in enumerate(series):
        row = i // max_images_per_row  # Determine which row to place the image
        column = i % max_images_per_row  # Determine which column to place the image

        img_array = normalize_image(img_array)
        img = Image.fromarray(img_array)
        img.thumbnail((100, 100))
        img_tk = ImageTk.PhotoImage(img)
        photo_images.append(img_tk)

        btn = tk.Button(window, image=img_tk, command=lambda i=i: on_selection_clicked(i, window, image_sets, selected_indices, root, set_len, phase, set_index, phases, name_xml))
        btn.image = img_tk  # Keep a reference
        btn.grid(row=row, column=column)

    window.geometry("+100+100")  

    # Buttons for skipping phase and marking blurry
    skip_btn = tk.Button(window, text="Skip Phase", command=lambda: on_skip_clicked(window, image_sets, selected_indices, root, phase, set_index, phases, name_xml))
    skip_btn.grid(row=(set_len // max_images_per_row + 1), column=0, sticky='ew')

    blurry_btn = tk.Button(window, text="Mark as Blurry", command=lambda: on_blurry_clicked(window, image_sets, selected_indices, root, set_index, phases, name_xml))
    blurry_btn.grid(row=(set_len // max_images_per_row + 1), column=1, sticky='ew')

    resume_btn = tk.Button(window, text="Resume", command=lambda: on_resume_clicked(window, image_sets, selected_indices, root, set_index, phases, name_xml))
    resume_btn.grid(row=(set_len // max_images_per_row + 1), column=4, sticky='ew')

    # Back Button
    if selected_indices:
    # if set_index > 0 or (set_index == 0 and len(selected_indices[set_index]) > 1):
        back_btn = tk.Button(window, text="Back", command=lambda: go_back(window, image_sets, selected_indices, root, phase, set_index, phases, name_xml))
        back_btn.grid(row=(set_len // max_images_per_row + 1), column=2, sticky='ew')

        save_btn = tk.Button(window, text="Save", command=lambda: on_save_clicked(selected_indices, phases, name_xml))
        save_btn.grid(row=(set_len // max_images_per_row + 1), column=3, sticky='ew')

def on_resume_clicked(window,image_sets, selected_indices, root, set_index, phases, name_xml):
    """Resumes UI based on what is in name_xml

    Args:
        window (tk.window): Current UI window
        image_sets (_type_): _description_
        selected_indices (list): list of dicts which contain 
        root (tk.root): root of TK
        set_index (_type_): _description_
        phases (list): List of strings for phases
        name_xml (string): Path to Name XML
    """
    stored_selections = parse_xml_for_phases_resume(name_xml)
    print(stored_selections)
    
    for stored in stored_selections.values():
        print(stored)
        selected_indices.append(stored)
        set_index+=1


    window.destroy()
    display_set(image_sets, set_index, selected_indices, root, phases[0], phases, name_xml)

def on_save_clicked(selected_indices, phases, name_xml):
    
    store_results_multiclass(images_dict, selected_indices, name_xml, phases)

# clicks and sets output
def on_selection_clicked(index, window, image_sets, selected_indices, root, set_len, phase, set_index, phases, name_xml):
    #  creates new dict if not at the end
    if len(selected_indices) <= set_index:
        selected_indices.append({})

    selected_indices[set_index][phase] = index
    print(f"Selected {phase} in set {set_index + 1}: {index}")
    
    window.destroy()
    handle_next_phase_or_set(window,image_sets, selected_indices, root, phase, set_index, phases, name_xml)

def handle_next_phase_or_set(window,image_sets, selected_indices, root, phase, set_index, phases, name_xml):
    
    next_index = phases.index(phase) + 1
    if next_index < len(phases):
        display_set(image_sets, set_index, selected_indices, root, phases[next_index], phases, name_xml)
    else:
        if set_index + 1 < len(image_sets):
            display_set(image_sets, set_index + 1, selected_indices, root, phases[0], phases, name_xml)
        else:
            messagebox.showinfo("Completed", "All selections completed.")
            print("Final selections:", selected_indices)
            root.quit()
            root.destroy()

def on_blurry_clicked(window, image_sets, selected_indices, root, set_index, phases, name_xml):
    if len(selected_indices) <= set_index:
        selected_indices.append({phase: 'blurry' for phase in phases})

    window.destroy()
    if set_index + 1 < len(image_sets):
        display_set(image_sets, set_index + 1, selected_indices, root, phases[0], phases, name_xml)
    else:
        messagebox.showinfo("Completed", "All selections completed.")
        root.quit()
        root.destroy()

def on_skip_clicked(window, image_sets, selected_indices, root, phase, set_index, phases, name_xml):
    if len(selected_indices) <= set_index:
        selected_indices.append({})

    selected_indices[set_index][phase] = 'skipped'
    print(f"Skipped {phase} in set {set_index + 1}")
    window.destroy()
    handle_next_phase_or_set(window, image_sets, selected_indices, root, phase, set_index, phases, name_xml)

def go_back(window, image_sets, selected_indices, root, phase, set_index, phases, name_xml):
    if set_index > 0 or (set_index == 0 and len(selected_indices[set_index]) > 1):
        # Revert to previous phase or image set
        previous_phase_index = phases.index(phase) - 1
        if previous_phase_index >= 0:
            # Go back within the same set
            display_set(image_sets, set_index, selected_indices, root, phases[previous_phase_index], phases, name_xml)
        else:
            # Go back to the previous set
            if set_index > 0:
                display_set(image_sets, set_index - 1, selected_indices, root, phases[-1], phases, name_xml)
    window.destroy()

        


def load_ui(cell_xml):
    """Entry point, loads the name selector. Then gets all the images fo cells and
    returns them as a dict. Then stores the results.

    Args:
        cell_xml (string): Path to cell_reigons.xml
    """
    # Call the UI function to run the name selector
    
    name_xml = run_name_selector("select_xmls")
    print(f"Selected XML: {name_xml}")
    images_dict = get_all_images(cell_xml)
    selected_indicies = load_selector(images_dict, 0)
    store_results(images_dict, selected_indicies, name_xml)

def load_ui_from_folder(phases, verbose = False):
    """Entry point, loads a folder selector then the name selector. 
    Normalises paths.
    Gets all the images fo cells and returns them as a dict. 
    Then stores the results.

    Args:
        phases (list): List of strings for phases
        verbose (bool): Decides whether to print variables
    """
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
    global images_dict
    images_dict = get_all_images(cell_xml)
    if verbose:
        print(images_dict)
    selected_indicies = load_selector(images_dict, 0, phases, name_xml)
    if verbose:
        print(selected_indicies)
    store_results_multiclass(images_dict, selected_indicies, name_xml, phases)
    
def xml_to_labels(name_xml, cell_xml):
    """Coverts for single phase.

    Args:
        name_xml (string): Path to user selections CML
        cell_xml (string): Path to cell_reigons.xml
    """
    
    user_df = read_xml_to_dataframe(name_xml)
    
    cell_df = cell_xml_to_dataframe(cell_xml)
    
    target_class_id = 2
    modified_df = modify_class_ids(cell_df, user_df, target_class_id)
    append_modified_labels(modified_df)
    
    
def debug_xml_to_labels(name_xml, cell_xml):
    """ Debugger to view single phase.

    Args:
        name_xml (string): Path to user selections CML
        cell_xml (string): Path to cell_reigons.xml

    Returns:
        pd.df, pd.df: Pair of dataframes for selector and clicker XML
    """
    
    user_df = read_xml_to_dataframe(name_xml)
    
    cell_df = cell_xml_to_dataframe(cell_xml)
    return user_df, cell_df

