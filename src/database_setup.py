import pandas as pd
from sqlalchemy import create_engine, text
import json

# Database connection
DATABASE_URI = 'mysql+mysqlconnector://root:200103@localhost:3306/akinator'

# Create SQLAlchemy engine 
engine = create_engine(DATABASE_URI)

# Load the cleaned data file
df = pd.read_csv('./data/cleaned_data.csv')

# Check if data is loaded correctly
print("Data Loaded: ", df.head())  

with engine.connect() as connection:
    # Create the table if it doesn't exist
    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS person (
            id INT AUTO_INCREMENT PRIMARY KEY,
            character_name VARCHAR(255) NOT NULL,
            category VARCHAR(255),
            traits JSON
        )
    """))
    
    # Insert data into the table
    for _, row in df.iterrows():
        # Print each row to ensure data is correct
        print(f"Inserting: {row['character_name']}, {row['category']}, {row['traits']}")
        
        connection.execute(
            text("""
                INSERT INTO person (character_name, category, traits)
                VALUES (:character_name, :category, :traits)
            """),
            {
                "character_name": row['character_name'],
                "category": row['category'],
                "traits": json.dumps(row['traits'])  # Convert traits to JSON 
            }
        )
    
    # Commit the transaction 
    connection.commit()

print("Data insertion completed.")
