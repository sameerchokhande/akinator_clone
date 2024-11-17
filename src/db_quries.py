import pandas as pd
from sqlalchemy import create_engine, text
import json

 
DATABASE_URI = 'mysql+mysqlconnector://root:200103@localhost:3306/akinator'
engine = create_engine(DATABASE_URI)

#  fetch all characters
def fetch_all_characters():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT * FROM person"))  
        characters = [dict(row) for row in result]
    return characters

#  filter characters by traits
def fetch_characters_by_traits(traits):
    with engine.connect() as connection:
        query = text("""
            SELECT * FROM person  
            WHERE JSON_CONTAINS(traits, :traits)
        """)
        result = connection.execute(query, {"traits": json.dumps(traits)})
        characters = [dict(row) for row in result]
    return characters
