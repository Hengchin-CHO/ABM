import os
import json
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

def read_json(file_path):
    """Read JSON file and return the 'Landscape' data."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data["Landscape"]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None

# Function to calculate statistics for a single file
def calculate_stats(data):
    max_val = np.max(data)
    min_val = np.min(data)
    mean_val = np.mean(data)
    std_dev = np.std(data)
    kurt = kurtosis(data)
    skw = skew(data)
    return max_val, min_val, mean_val, std_dev, kurt, skw

# Path to the folder containing the JSON files
folder_path = 'landscape/'

# List to store the results temporarily
results = []

# Traverse the directory structure
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(root, file)
            data = read_json(file_path)
            if data is not None:
                max_val, min_val, mean_val, std_dev, kurt, skw = calculate_stats(data)
                land_number = os.path.splitext(file)[0]  # Get the file name without extension
                results.append({
                    "land_number": land_number,
                    "max": max_val,
                    "min": min_val,
                    "mean": mean_val,
                    "std_dev": std_dev,
                    "kurtosis": kurt,
                    "skew": skw,
                    #"folder": os.path.basename(root)
                })

# Convert the list of dictionaries to a DataFrame
stats_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
csv_path = "landscape_statistics.csv"
stats_df.to_csv(csv_path, index=False)

print(f"Statistics saved to {csv_path}")
