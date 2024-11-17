from sqlalchemy import create_engine
import json

# Database connection URI for MySQL
DATABASE_URI = 'mysql+mysqlconnector://root:200103@localhost:3306/akinator'

# Connect to the database
engine = create_engine(DATABASE_URI)

def fetch_all_characters():
    query = "SELECT character_name, traits FROM person"
    with engine.connect() as connection:
        result = connection.execute(query)
        characters = [
            {
                "character_name": row["character_name"],
                "traits": json.loads(row["traits"])
            }
            for row in result
        ]
    return characters
