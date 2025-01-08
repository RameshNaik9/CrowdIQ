import firebase_admin
from firebase_admin import credentials, firestore
import json

# Initialize Firestore
def initialize_firestore(service_account_path):
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred)
    return firestore.client()

# Recursively retrieve all documents in all collections
def fetch_all_documents(db):
    def fetch_collection(collection_ref, data):
        for doc in collection_ref.stream():
            doc_data = doc.to_dict()
            doc_data["_id"] = doc.id  # Add document ID
            data.append(doc_data)

            # Recursively fetch subcollections
            subcollections = doc.reference.collections()
            for subcollection in subcollections:
                sub_data = []
                fetch_collection(subcollection, sub_data)
                doc_data[subcollection.id] = sub_data

    all_data = {}
    collections = db.collections()
    for collection in collections:
        collection_name = collection.id
        collection_data = []
        fetch_collection(collection, collection_data)
        all_data[collection_name] = collection_data
    return all_data

# Save data to a JSON file
def save_to_json_file(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Main function
def main():
    service_account_path = "D:\msc\Virture.json"  # Replace with your path
    output_file = "firestore_data.json"  # Output file for Firestore data

    db = initialize_firestore(service_account_path)
    print("Fetching all documents from Firestore...")
    data = fetch_all_documents(db)
    print("Saving data to JSON file...")
    save_to_json_file(data, output_file)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
