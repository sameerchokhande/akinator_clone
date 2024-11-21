import pandas as pd

try:
    print("Starting the script...")

    # Step 1: Load the dataset
    file_path = "data/row_data.csv"  # Update the path if necessary
    print(f"Loading dataset from: {file_path}")
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!")

    # Step 2: Extract the first 500 rows
    print("Extracting the first 500 rows...")
    completed_data = data.head(500)

    # Step 3: Save the extracted rows to a new CSV file
    output_file_path = "data/completed_500_characters.csv"
    print(f"Saving the extracted data to: {output_file_path}")
    completed_data.to_csv(output_file_path, index=False)
    print("File saved successfully!")

except Exception as e:
    print(f"An error occurred: {e}")
