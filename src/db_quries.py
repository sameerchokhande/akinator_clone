import pandas as pd
from sqlalchemy import create_engine
import ast
from sqlalchemy.sql import text
import json

# Database connection URI for MySQL
DATABASE_URI = 'mysql+mysqlconnector://root:200103@localhost:3306/akinator'

# Connect to the database
engine = create_engine(DATABASE_URI)


def fetch_all_characters():
    """
    Fetch all characters from the database and parse their traits.
    """
    query = text("SELECT character_name, traits FROM person")  # Wrap the query in text()
    with engine.connect() as connection:
        result = connection.execute(query).mappings().fetchall()  # Use mappings() for dict-like rows

    characters = []
    for row in result:
        character_name = row['character_name']  # Access by key since rows are dicts
        try:
            # Parse traits from JSON string
            traits = json.loads(row['traits'])
            characters.append({'character_name': character_name, 'traits': traits})
        except json.JSONDecodeError:
            print(f"Error decoding traits for {character_name}: {row['traits']}")

    # Debugging: Check loaded characters
    # print("Loaded Characters:", characters)
    return characters

# Fetch characters from the database
characters = fetch_all_characters()

if not characters:
    print("No characters were loaded from the database.")
else:
    print("Characters successfully loaded.")
