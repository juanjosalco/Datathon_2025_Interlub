import pandas as pd

# Load the CSV file
file_path = "/Users/juansalazar/Documents/Datathon_2025_Interlub/utils/Datathon.csv"
df = pd.read_csv(file_path)

# Convert to JSON format
json_data = df.to_json(orient="records", indent=4)

print(json_data)