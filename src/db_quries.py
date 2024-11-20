import pandas as pd
from sqlalchemy import create_engine, text
import json

DATABASE_URI = 'mysql+mysqlconnector://root:200103@localhost:3306/akinator'
engine = create_engine(DATABASE_URI)

# Fetch all characters
def fetch_all_characters():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT * FROM person"))
            # Convert the result to a list of dictionaries
            characters = [dict(row) for row in result.fetchall()]  # Ensure fetchall is used
        return characters
    except Exception as e:
        print(f"Error fetching all characters: {e}")
        return []

# Filter characters by traits
def fetch_characters_by_traits(traits):
    try:
        with engine.connect() as connection:
            # Ensure traits is a valid JSON object before passing it to the query
            if isinstance(traits, dict):
                traits = json.dumps(traits)
            query = text("""
                SELECT * FROM person  
                WHERE JSON_CONTAINS(traits, :traits)
            """)
            result = connection.execute(query, {"traits": traits})
            # Convert the result to a list of dictionaries
            characters = [dict(row) for row in result.fetchall()]  # Ensure fetchall is used
        return characters
    except Exception as e:
        print(f"Error filtering characters by traits: {e}")
        return []
characters = fetch_all_characters()
print(characters)