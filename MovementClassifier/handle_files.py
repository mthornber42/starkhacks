import json
import pandas as pd

def json_to_csv(json_file):
    # Open the file and load the list of objects
    with open(json_file, 'r') as file:
        data = json.load(file)

    if "samples" in data:
        data = data["samples"]
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(json_file.replace('.json', '.csv'), index=False, header=False)

def read_csv(csv_file):
    df = pd.read_csv(csv_file, header=None)
    return df.to_numpy()

def save_csv(arr, csv_file):
    df = pd.DataFrame(arr)
    df.to_csv(csv_file, index=False, header=False)

