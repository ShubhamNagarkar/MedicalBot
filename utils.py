import json


def read_json(file_path):
    """
    This function  reads contents of json file into a dictionary.
    """
    data = {}
    
    with open(file_path, 'r') as file:
        data = json.load(file)

    return data


def write_json(file_path, data):
    """
    This function writes a dictionary to a json file.
    """
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

