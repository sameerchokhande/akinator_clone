import json
from sqlalchemy import create_engine
from sqlalchemy.sql import text

# Database connection
engine = create_engine('mysql+pymysql://root:200103@localhost:3306/akinator')

import re

def fix_invalid_json(traits):
    # Replace single quotes with double quotes and add commas if necessary
    traits = traits.replace("'", '"')
    traits = re.sub(r'"\s+"', '", "', traits)  # Fix missing commas
    return traits

def fix_traits():
    query_select = text("SELECT id, traits FROM person")
    query_update = text("UPDATE person SET traits = :traits WHERE id = :id")

    with engine.connect() as connection:
        result = connection.execute(query_select).fetchall()
        for row in result:
            id = row[0]
            raw_traits = row[1]
            try:
                # Fix and parse traits
                cleaned_traits = fix_invalid_json(raw_traits.strip())
                clean_traits = json.loads(cleaned_traits)
                connection.execute(query_update, {"id": id, "traits": json.dumps(clean_traits)})
            except Exception as e:
                print(f"Error cleaning traits for ID {id}: {e}")
                print(f"Raw traits: {raw_traits}")

fix_traits()


