import joblib
from db_quries import fetch_all_characters
import pandas as pd
from model_training import load_character_data

# Load the trained model
model = joblib.load('src/question_selector_model.pkl')

# Load the trained model
model = joblib.load('src/question_selector_model.pkl')

def dynamic_question_selection(characters):
    """
    Selects the most effective trait (question) to ask based on the current state.
    """
    all_traits = set()
    for char in characters:
        all_traits.update(char['traits'])
    
    trait_df = pd.DataFrame([{trait: 1 if trait in char['traits'] else 0 for trait in all_traits} for char in characters])
    
    # Match training columns to the model's expected features
    expected_features = model.feature_names_in_
    for feature in expected_features:
        if feature not in trait_df.columns:
            trait_df[feature] = 0

    trait_df = trait_df[expected_features]

    # Debugging: Check trait DataFrame
    # print("Trait DataFrame for Prediction:")
    # print(trait_df.head())

    # Predict probabilities and select the most effective trait
    predictions = model.predict_proba(trait_df)
    predictions_mean = predictions.mean(axis=0)
    most_effective_trait_index = predictions_mean.argmax()
    most_effective_trait = expected_features[most_effective_trait_index]

    # Debugging: Check selected trait
    print(f"Selected Trait: {most_effective_trait}")
    return most_effective_trait


# Main game loop
def play_game():
    data = load_character_data()
    questions = list(data.columns[1:])
    characters = fetch_all_characters()
    if not characters:
        print("No characters were loaded from the database.")
        return

    while len(characters) > 1:
        trait = dynamic_question_selection(characters)
        for question in questions:
            print(f"Is the character {question.replace('_', ' ')}? (yes/no)")
        answer = input().strip().lower()

        if answer == 'yes':
            characters = [char for char in characters if trait in char['traits']]
        elif answer == 'no':
            characters = [char for char in characters if trait not in char['traits']]
        else:
            print("Please answer with 'yes' or 'no'.")
        # Game matching logic
        # matching_characters = [c for c in characters if all_trait in c['traits']]
        # if matching_characters:
        #     print(f"Matching Character: {matching_characters[0]['character_name']}")
        # else:
        #     print("No matching character found.")


    if len(characters) == 1:
        print(f"The character you're thinking of is: {characters[0]['character_name']}")
    else:
        print("No matching character found.")

if __name__ == "__main__":
    play_game()
