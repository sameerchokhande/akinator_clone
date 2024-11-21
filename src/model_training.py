import pandas as pd
from sqlalchemy import create_engine
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Database connection URI for MySQL
DATABASE_URI = 'mysql+mysqlconnector://root:200103@localhost:3306/akinator'

# Connect to the database
engine = create_engine(DATABASE_URI)

def load_character_data():
    # Query character data from the database
    query = "SELECT character_name, traits FROM person"
    with engine.connect() as connection:
        data = pd.read_sql(query, connection)

    # Convert stringified list to an actual Python list
    def parse_traits(traits_string):
        try:
            # Use `ast.literal_eval` to safely evaluate the string
            parsed_traits = ast.literal_eval(traits_string)
            # Flatten the nested structure if needed and remove quotes/spaces
            return [trait.strip(" '\"") for trait in parsed_traits]
        except Exception as e:
            print(f"Error parsing traits: {traits_string} - {e}")
            return []
    
    # Apply parsing function
    data['parsed_traits'] = data['traits'].apply(parse_traits)

    # Create one-hot encoded columns for traits
    all_traits = set(trait for traits in data['parsed_traits'] for trait in traits)
    for trait in all_traits:
        data[trait] = data['parsed_traits'].apply(lambda x: 1 if trait in x else 0)

    # print("Expanded Data Columns:", data.columns)
    print("Expanded Data Sample:")
    print(data.head())
    return data.drop(columns=['traits', 'parsed_traits'])

# Load data from the database
data = load_character_data()

def create_training_data(data):
    """
    Create a DataFrame for training the model, where each row represents a question 
    for a specific character.
    """
    questions = list(data.columns[1:])  # Trait names as questions
    training_rows = []  # Collect rows as dictionaries

    for question in questions:
        for _, row in data.iterrows():
            if question in data.columns:  # Ensure the trait exists in the data
                training_rows.append({
                    'question': question,
                    'character_name': row['character_name'],
                    'effective': row[question]  # Value of the trait (1 or 0)
                })

    # Convert rows to a DataFrame
    training_data = pd.DataFrame(training_rows)
    # print("Training Data Columns:", training_data.columns)
    print("Training Data Sample:")
    print(training_data.head())
    return training_data

# Generate training data
training_data = create_training_data(data)

# Ensure the 'question' column exists
if 'question' not in training_data.columns:
    raise ValueError("The 'question' column is missing in training_data.")

# Convert training data to numeric for the model
X = pd.get_dummies(training_data[['question']])  # One-hot encode the 'question' column
y = training_data['effective']

# Debugging output for one-hot encoding
# print("One-Hot Encoded Feature Matrix (X):")
# print(X.head())

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model to a file
joblib.dump(model, 'src/question_selector_model.pkl')
print("Model saved successfully!")
