import xml.etree.ElementTree as ET
import pandas as pd


def update_xml(file_name, series_id, selected_index, xml_file_name):
    # Parse the existing XML file
    tree = ET.parse(xml_file_name)
    root = tree.getroot()

    # Search for an existing entry with the same "File" and "SeriesID"
    for data_entry in root.findall('DataEntry'):
        file_element = data_entry.find('PathName')
        series_id_element = data_entry.find('SeriesID')

        # Check if the file and series_id match
        if file_element is not None and series_id_element is not None and \
           file_element.text == file_name and series_id_element.text == str(series_id):
            # Update the selected_index value
            selected_index_element = data_entry.find('SelectedIndex')
            selected_index_element.text = str(selected_index)
            break
    else:
        # If no matching entry was found, create a new one
        data_entry = ET.Element('DataEntry')
        file_element = ET.SubElement(data_entry, 'PathName')
        series_id_element = ET.SubElement(data_entry, 'SeriesID')
        selected_index_element = ET.SubElement(data_entry, 'SelectedIndex')

        file_element.text = file_name
        series_id_element.text = str(series_id)
        selected_index_element.text = str(selected_index)

        root.append(data_entry)

    # Save the updated XML to the file
    tree.write(xml_file_name)


def store_results(images_dict, selected_indicies, xml_file_name):
    for (file_name, series_id ), selected_index in zip(images_dict.keys(), selected_indicies):
        print(file_name, series_id, selected_index, xml_file_name)
        update_xml(file_name, series_id, selected_index, xml_file_name)    


def read_xml_to_dataframe(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Create a list to hold the data
    data_entries = []

    # Iterate through each 'DataEntry' in the XML
    for data_entry in root.findall('DataEntry'):
        file = data_entry.find('PathName').text
        print(file)
        series_id = data_entry.find('SeriesID').text
        selected_index = data_entry.find('SelectedIndex').text

        # Append this entry as a dict to the list
        data_entries.append({
            'PathName': file,
            'SeriesID': int(series_id),
            'SelectedIndex': int(selected_index)
        })

    # Create a DataFrame from the list of dicts
    df = pd.DataFrame(data_entries)
    return df


