import joblib
from db_quries import fetch_all_characters
import pandas as pd
from model_training import load_character_data

# Load the trained model
model = joblib.load('src/question_selector_model.pkl')

def dynamic_question_selection(characters):
    # Define characters or patterns to ignore
    ignore = ['"', "'", "question_" ]  # Add any unwanted symbols here

    all_traits = set()
    for char in characters:
        all_traits.update(char['traits'])

    trait_df = pd.DataFrame([
        {trait: 1 if trait in char['traits'] else 0 for trait in all_traits}
        for char in characters
    ])

    # Ensure the DataFrame has the same features as the model's training data
    expected_features = model.feature_names_in_
    for feature in expected_features:
        if feature not in trait_df.columns:
            trait_df[feature] = 0

    trait_df = trait_df[expected_features]

    predictions = model.predict_proba(trait_df)
    predictions_mean = predictions.mean(axis=0)
    most_effective_trait_index = predictions_mean.argmax()
    most_effective_trait = expected_features[most_effective_trait_index]

    # Clean the selected trait using the ignore list
    for unwanted in ignore:
        most_effective_trait = most_effective_trait.replace(unwanted, "")

    # Replace underscores with spaces for better readability
    return most_effective_trait.replace("_", " ")

def play_game():
    data = load_character_data()
    characters = fetch_all_characters()
    
    if not characters:
        print("No characters loaded from the database.")
        return

    while len(characters) > 1:
        most_effective_trait = dynamic_question_selection(characters)
        print(f"Is the character {most_effective_trait}? (yes/no)")
        
        answer = input().strip().lower()

        # Normalize the trait for comparison
        matching_trait = most_effective_trait.replace(" ", "_").lower()

        print(f"Matching trait: {matching_trait}")  # Debugging: Check the normalized trait

        if answer == 'yes':
            characters = [
                char for char in characters 
                if matching_trait in [trait.strip().lower() for trait in char['traits']]
            ]
        elif answer == 'no':
            characters = [
                char for char in characters 
                if matching_trait not in [trait.strip().lower() for trait in char['traits']]
            ]
        else:
            print("Please answer with 'yes' or 'no'.")

        print(f"Remaining characters: {[char['character_name'] for char in characters]}")

    if len(characters) == 1:
        print(f"The character you're thinking of is: {characters[0]['character_name']}")
    else:
        print("No matching character found.")

if __name__ == "__main__":
    play_game()
