import json
from sqlalchemy import create_engine, text

# Database connection URI for MySQL
DATABASE_URI = 'mysql+mysqlconnector://root:200103@localhost:3306/akinator'
engine = create_engine(DATABASE_URI)

def clean_and_fix_traits():
    """
    Fix the traits in the database to ensure consistent JSON array format.
    """
    with engine.connect() as connection:
        # Fetch all records
        result = connection.execute(text("SELECT id, traits FROM person")).fetchall()
        
        for row in result:
            record_id, raw_traits = row
            try:
                # Attempt to parse the current traits
                traits = json.loads(raw_traits)
                if not isinstance(traits, list):
                    raise ValueError("Traits must be a list.")
                # Re-encode to ensure JSON format
                clean_traits = json.dumps(traits)
            except Exception as e:
                print(f"Error cleaning traits for ID {record_id}: {raw_traits} - {e}")
                clean_traits = "[]"  # Default to an empty list on error

            # Update the database with cleaned traits
            connection.execute(
                text("UPDATE person SET traits = :clean_traits WHERE id = :id"),
                {"clean_traits": clean_traits, "id": record_id}
            )
        connection.commit()

clean_and_fix_traits()
print("Database traits have been cleaned and standardized.")
