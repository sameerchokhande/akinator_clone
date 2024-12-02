import json
from sqlalchemy import create_engine
from sqlalchemy.sql import text

# Database connection URI for MySQL
DATABASE_URI = 'mysql+mysqlconnector://root:200103@localhost:3306/akinator'

# Connect to the database
engine = create_engine(DATABASE_URI)

# Define the relevant traits (you can add or remove as needed)
RELEVANT_TRAITS = set([
    'is_male', 'is_female', 'is_freedom_fighter', 'is_non_violent', 'is_humble',
    'is_selfless', 'is_truthful', 'is_brave]' ,'is_artist', 'is_writer', 'is_painter',
    'is_nobel_prize_winner', 'is_scientist', 'is_11th_president_of_india', 'is_missile_man_of_india',
    'is_visionary_leader', 'is_politician', 'is_cricketer', 'is_god_of_cricket', 'is_boxer',
    'is_chess_player', 'is_athlete', 'is_footballer', 'is_actor', 'is_big_b', 'is_politician'
])


def fetch_all_characters():
    # Query to fetch character name and traits from the database
    query = text("SELECT character_name, traits FROM person")
    
    with engine.connect() as connection:
        result = connection.execute(query).mappings().fetchall()

    characters = []
    for row in result:
        character_name = row['character_name']
        traits_str = row['traits']
        
        # Split traits string into a list and clean each trait
        traits = [trait.strip().strip('"') for trait in traits_str.split(',')] if traits_str else []
        
        # print(f"Raw traits for {character_name}: {traits}")
        
        # Filter traits based on relevant traits
        filtered_traits = [trait for trait in traits if trait.lower() in RELEVANT_TRAITS]
        # print(filtered_traits)
        if filtered_traits:
            characters.append({'character_name': character_name, 'traits': filtered_traits})
        else:
            print(f"Skipping {character_name} as no relevant traits were found.")

    # print("Loaded Characters:", characters)
    return characters


# Fetch characters from the database
characters = fetch_all_characters()
# print(characters)
if not characters:
    print("No characters were loaded from the database.")
else:
    print("Characters successfully loaded.")
