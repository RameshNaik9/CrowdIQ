import csv
from firebase_admin import credentials, firestore
import threading
from threading import Thread
from utils1 import os, datetime

csv_lock = threading.Lock()
final_date = '2025-04-15' 

def create_csv(csv_file):
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['serial_number', 'tracking_id', 'gender', 'time_spent', 'start_time', 'first_appearance_time', 'last_appearance_time'])

def db_add_new(db,data,date_str):
    # Create a new collection for the current date
    collection_name = f'Visiter-Virtue-{date_str}'
    try:
        doc_ref = db.collection(collection_name).document(str(data['serial_number']))
        doc_ref.set(data)
    except Exception as e:
        print(f"Error adding data to Firestore: {e}")

def db_update_time(db,serial_number, new_time_spent, last_appearance_time,date_str):
    # Update document in the collection for the current date
    collection_name = f'Visiter-Virtue-{date_str}'
    try:
        doc_ref = db.collection(collection_name).document(str(serial_number))
        doc_ref.update({
            'time_spent': new_time_spent,
            'last_appearance_time': last_appearance_time
        })
    except Exception as e:
        print(f"Error updating time in Firestore: {e}")


def add_new_entry_to_csv(csv_file,track_info,track_id):
    """Add a new row for a unique tracking_id with gender, age, and appearance times only once."""
    with csv_lock:  # Lock the CSV file for exclusive access
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            data = track_info[track_id]
            writer.writerow([
                data['serial_number'], track_id,
                data['gender'], data['age'], data['time_spent'],
                data['start_time'], data['first_appearance_time'], data['last_appearance_time']
            ])
def update_time_spent_in_csv(csv_file,serial_number, new_time_spent):
    """Update the time_spent and last_appearance_time fields for a given serial_number in the CSV file."""
    rows = []

    with csv_lock:  # Lock the CSV file for exclusive access
        if os.path.getsize(csv_file) == 0:
            print("CSV file is empty; cannot update time.")
            return

        # Read all rows into memory
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.reader(file)

            try:
                headers = next(reader)  # Read headers
            except StopIteration:
                print("No headers found; exiting update.")
                return

            for row in reader:
                if int(row[0]) == serial_number:
                    row[4] = f"{new_time_spent:.2f}"  # Update time_spent field
                    row[7] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Update last_appearance_time
                rows.append(row)

        # Write updated rows back to the CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)  # Write headers
            writer.writerows(rows)    # Write all rows, including modified ones



def async_update_db_and_csv(db,csv_file,track_data, elapsed_time,date_str):
    # Update CSV
    print("Time : ",elapsed_time)
    update_time_spent_in_csv(csv_file,track_data['serial_number'], elapsed_time)
    # Update Firebase with new time spent and last appearance
    db_update_time(db,track_data['serial_number'], elapsed_time, track_data['last_appearance_time'],date_str)
