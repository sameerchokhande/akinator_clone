import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import ast
# Database connection URI for MySQL
DATABASE_URI = 'mysql+mysqlconnector://root:200103@localhost:3306/akinator'

# Connect to the database
engine = create_engine(DATABASE_URI)


def clean_traits(traits_string):
    """
    Cleans and parses the traits string from the database.
    - Removes unwanted characters and converts it to a list of traits.
    """
    try:
        # Convert the string to a Python object (list of traits)
        traits_list = ast.literal_eval(traits_string)
        # Flatten the list (if nested) and strip any extra characters or spaces
        clean_list = [trait.strip() for trait in traits_list]
        return clean_list
    except Exception as e:
        print(f"Error while parsing traits: {traits_string} - {e}")
        return []


def load_character_data():
    """
    Load and preprocess character data from the database.
    - Expands traits into one-hot encoded columns.
    """
    query = "SELECT character_name, traits FROM person"
    with engine.connect() as connection:
        data = pd.read_sql(query, connection)

    print(f"Total rows fetched from database: {data.shape[0]}")  # Print total rows

    # Collect all unique traits
    all_traits = set(trait for traits_string in data['traits'] for trait in traits_string.split(','))

    # Create one-hot encoded columns for traits
    for trait in all_traits:
        data[trait.strip()] = data['traits'].apply(lambda x: 1 if trait in x else 0)

    # Display the full dataset
    # print("Expanded Data Preview:")
    # print(data)

    # Drop the original traits column and return the processed DataFrame
    return data.drop(columns=['traits'])


def create_training_data(data):
    """
    Create a DataFrame for training the model.
    - Each row represents a question for a specific character.
    """
    questions = [col.replace("question", "").lower() for col in data.columns[1:]]  # Remove prefix
    training_rows = []  # Collect rows as dictionaries

    for question in questions:
        for _, row in data.iterrows():
            training_rows.append({
                'question': question,
                'character_name': row['character_name'],
                'effective': row[question]  # Value of the trait (1 or 0)
            })

    training_data = pd.DataFrame(training_rows)
    return training_data



# Step 1: Load and preprocess data
data = load_character_data()

# Step 2: Generate training data
training_data = create_training_data(data)

# Step 3: Ensure the 'question' column exists
if 'question' not in training_data.columns:
    raise ValueError("The 'question' column is missing in training_data.")

# Step 4: Convert training data to numeric
X = pd.get_dummies(training_data[['question']])  # One-hot encode the 'question' column
y = training_data['effective']

# Debugging output for one-hot encoding
# print("One-Hot Encoded Feature Matrix (X):")
# print(X)

# Step 5: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 8: Save the trained model to a file
joblib.dump(model, 'src/question_selector_model.pkl')
print("Model saved successfully!")

# Step 9: Verify distribution of the `effective` column
# print("Distribution of 'effective' in training data:")
# print(training_data['effective'].value_counts())
