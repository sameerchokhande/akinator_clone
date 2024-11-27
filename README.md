Akinator Clone
This is a Python-based Akinator-inspired project that identifies a character by asking a series of yes/no questions. It uses a MySQL database for storing characters and their traits, Random Forest Classifier for training the model, and provides an interactive gameplay experience.
Features
Reads character traits from a MySQL database.
Converts traits into a one-hot-encoded dataset for machine learning.
Trains a model to guess the character based on yes/no answers.
Achieves high model accuracy with Random Forest Classifier.
Saves the trained model for reuse in gameplay.

Setup Instructions
1. Clone the Repository
git clone https://github.com/your-username/akinator-clone.git
cd akinator-clone
2. Install Dependencies
Ensure Python 3.8 or higher is installed. Install the required libraries using the following command:

pip install -r requirements.txt
3. Create MySQL Database
Create a MySQL database named akinator.
Import your character data into a table named person with the following columns:
character_name (VARCHAR)
traits (TEXT)
4. Set up the Database URI
Update the DATABASE_URI in the script with your MySQL credentials:

DATABASE_URI = 'mysql+mysqlconnector://<username>:<password>@<host>:<port>/<database>'
For example:

DATABASE_URI = 'mysql+mysqlconnector://root:password@localhost:3306/akinator'
5. Train the Model
Run the script to train the model and save it:

python src/train_model.py
This will:

Load data from the database.
Preprocess traits and generate training data.
Train a Random Forest Classifier.
Save the model as src/question_selector_model.pkl.
6. Play the Game
Use the gameplay script to guess characters:

python src/play_game.py
Follow the prompts to answer yes/no questions.

Libraries Used
Below is the list of libraries used in this project:

SQLAlchemy: For interacting with the MySQL database.
MySQL Connector: Backend driver for MySQL.
Pandas: For data manipulation and preprocessing.
Scikit-learn: For building and evaluating the machine learning model.
Joblib: For saving and loading the trained model.
AST: For parsing stringified data safely.
Installation Commands
Install the libraries using the following commands:

pip install sqlalchemy
pip install mysql-connector-python
pip install pandas
pip install scikit-learn
pip install joblib   
Project Structure
akinator-clone/
├── src/
│   ├── train_model.py       # Script to train and save the model
│   ├── play_game.py         # Gameplay script
│   ├── question_selector_model.pkl # Saved model
├── requirements.txt         # List of dependencies
├── README.md                # Documentation   

Contact
For any questions or feedback, feel free to contact:

Name: chokhande sameer balasaheb
Email: sameerchokhande2@example.com
GitHub: sameerchokhande
