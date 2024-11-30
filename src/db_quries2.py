from sqlalchemy import create_engine, text
import json

# Database connection URI for MySQL
DATABASE_URI = 'mysql+mysqlconnector://root:200103@localhost:3306/akinator'

# Connect to the database
engine = create_engine(DATABASE_URI)

def fetch_all_characters():
    query = text("SELECT character_name, traits FROM person")  # Wrap the query with text()
    with engine.connect() as connection:
        result = connection.execute(query)
        characters = [
            {
                "character_name": row["character_name"],
                "traits": json.loads(row["traits"]) if row["traits"] else {}
            }
            for row in result.mappings()  # Use mappings() to get dictionary-like rows
        ]
    return characters

# Example usage
characters = fetch_all_characters()
print(characters)