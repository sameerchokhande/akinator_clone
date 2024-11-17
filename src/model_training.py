import pandas as pd
from sqlalchemy import create_engine
import json
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

    # Convert traits JSON to a DataFrame
    traits_df = pd.json_normalize(data['traits'].apply(json.loads))
    data = pd.concat([data['character_name'], traits_df], axis=1)
    return data

# Load data from the database
data = load_character_data()
print("Data loaded and traits expanded:")
print(data.head())

# Create labeled examples based on traits
def create_training_data(data):
    questions = list(data.columns[1:])  # Trait names as questions
    training_data = [question]
    #print("Columns in data:", data.columns)


    for question in questions:
        for _, row in data.iterrows():
            training_data.append({
                'question': question,
                'character_name': row['character_name'],
                'effective': row[question]  # Check if the trait helps narrow down
            })
    if 'question' not in training_data.columns:
        print("Error: 'question' column missing in training_data.")
        print(training_data.columns)
    return pd.DataFrame(training_data)



# Generate training data
# training_data = create_training_data(data)
# print("Training data sample:")
# print(training_data.head())

# Convert training data to numeric
X = pd.get_dummies(training_data[['question']])
y = training_data['effective']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save the model to a file
joblib.dump(model, 'src/question_selector_model.pkl')
print("Model saved successfully!")
print(training_data.columns)
print(training_data.head())
