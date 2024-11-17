import joblib
from db_quries2 import fetch_all_characters
import pandas as pd

# Load the trained model
model = joblib.load('src/model_training.py')

# Function to dynamically select the best question
def dynamic_question_selection(characters):
    # Generate a DataFrame from remaining characters
    trait_df = pd.json_normalize([char['traits'] for char in characters])
    # Predict the best question
    predictions = model.predict_proba(trait_df)
    best_question = predictions.argmax()
    return trait_df.columns[best_question]

# Function to ask questions and filter characters dynamically
def ask_question(trait, characters):
    answer = input(f"Does the character have the trait '{trait}'? (yes/no): ").strip().lower()
    if answer == "yes":
        filtered = [char for char in characters if char['traits'].get(trait, False)]
    elif answer == "no":
        filtered = [char for char in characters if not char['traits'].get(trait, False)]
    else:
        print("Invalid input. Please answer 'yes' or 'no'.")
        return ask_question(trait, characters)

    print(f"Characters remaining after asking about '{trait}': {len(filtered)}")
    return filtered

# Main game loop
def play_game():
    characters = fetch_all_characters()  # Load all characters
    print("Loaded characters:", characters)  # Debugging line
    
    if not characters:
        print("No characters were loaded from the database.")
        return

    while len(characters) > 1:
        # Dynamically select the next question
        trait = dynamic_question_selection(characters)
        characters = ask_question(trait, characters)

    if len(characters) == 1:
        print(f"The character you're thinking of is: {characters[0]['character_name']}")
    elif len(characters) == 0:
        print("No matching character found.")
    else:
        print("Could not narrow down to a single character.")

if __name__ == "__main__":
    play_game()
