# import csv

# def filter_csv_columns_and_limit_records(input_file, output_file, limit=1000000):
#     # Define the columns you want to keep (based on header names)
#     columns_to_keep = ['artist_name', 'title', 'year', 'release']
    
#     # Open the input CSV file for reading
#     with open(input_file, 'r', encoding='utf-8') as infile:
#         reader = csv.DictReader(infile)
        
#         # Open the output CSV file for writing
#         with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
#             # Create a DictWriter with the filtered columns
#             writer = csv.DictWriter(outfile, fieldnames=columns_to_keep)
            
#             # Write the header with only the columns we want to keep
#             writer.writeheader()

#             # Iterate over each row and write only the filtered columns
#             row_count = 0
#             for row in reader:
#                 if row_count >= limit:
#                     break  # Stop after reaching the limit
                
#                 filtered_row = {key: row[key] for key in columns_to_keep if key in row}
#                 writer.writerow(filtered_row)
#                 row_count += 1

# # Specify the input CSV file path and the output CSV file path
# input_file = '/home/domas/Desktop/DAT640_MusicProject/DAT640-MusicCRS/databases/song_data.csv'
# output_file = '/home/domas/Desktop/DAT640_MusicProject/DAT640-MusicCRS/databases/song_data_filtered.csv'

# # Call the function to filter columns and limit to 1 million records
# filter_csv_columns_and_limit_records(input_file, output_file, limit=1000000)


# import pandas as pd
# import sqlite3

# genres = ["pop", "rock", "metal", "ballad", "brazilian kpop", "specialty", "baldy", "angry", "Sad", "Country", "Jazz", "hip hop", "reggae", "classical", "electronic", "indie", "folk", "blues", "r&b", "dance"]

# # Read the CSV file
# df = pd.read_csv("/home/domas/Desktop/DAT640_MusicProject/DAT640-MusicCRS/databases/song_data_filtered.csv")

# # Drop duplicate rows based on all columns
# # df = df.drop_duplicates()

# # Printing number of rows after dropping duplicates
# print("Number of rows after dropping duplicates: ", len(df))

# def insert_songs_in_bulk(songs):
#     """
#     Inserts multiple songs into the database using bulk insert.

#     Parameters:
#         songs (list of tuples): List of song details to be inserted.
#     """
#     conn = sqlite3.connect('/home/domas/Desktop/DAT640_MusicProject/DAT640-MusicCRS/databases/songs1.db')
#     cursor = conn.cursor()

#     try:
#         cursor.executemany('''
#             INSERT INTO songs (song_title, album_title, artist, release_date)
#             VALUES (?, ?, ?, ?)
#         ''', songs)
#         conn.commit()
#     except sqlite3.IntegrityError as e:
#         print(f"Error inserting song: {e}")
#     finally:
#         conn.close()

# # Prepare the songs data in a list of tuples
# songs = [
#     (df['title'][i], df['release'][i], df['artist_name'][i], df['year'][i])
#     for i in df.index
# ]

# # Batch size for inserting rows
# batch_size = 1000
# for i in range(0, len(songs), batch_size):
#     batch = songs[i:i + batch_size]
#     insert_songs_in_bulk(batch)

# print("Song data insertion complete.")
    
    
    

import pandas as pd
import sqlite3
import random

# List of genres
genres = ["pop", "rock", "metal", "ballad", "brazilian kpop", "specialty", "baldy", "angry", "Sad", 
          "Country", "Jazz", "hip hop", "reggae", "classical", "electronic", "indie", "folk", "blues", 
          "r&b", "dance"]

def modify_database_schema():
    """Add genre column to the songs table"""
    conn = sqlite3.connect('/home/domas/Desktop/DAT640_MusicProject/DAT640-MusicCRS/databases/songs1.db')
    cursor = conn.cursor()
    
    try:
        # Add new genre column
        cursor.execute('''
            ALTER TABLE songs
            ADD COLUMN genre TEXT
        ''')
        conn.commit()
        print("Successfully added genre column to the database.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("Genre column already exists.")
        else:
            print(f"Error modifying schema: {e}")
    finally:
        conn.close()

def update_genres_in_bulk():
    """Update all rows with random genres"""
    conn = sqlite3.connect('/home/domas/Desktop/DAT640_MusicProject/DAT640-MusicCRS/databases/songs1.db')
    cursor = conn.cursor()
    
    try:
        # Get all song IDs
        cursor.execute('SELECT song_id FROM songs')
        song_ids = cursor.fetchall()
        
        # Prepare updates with random genres
        updates = [(random.choice(genres), song_id[0]) for song_id in song_ids]
        
        # Update in batches
        batch_size = 1000
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            cursor.executemany('''
                UPDATE songs
                SET genre = ?
                WHERE song_id = ?
            ''', batch)
            conn.commit()
            print(f"Updated {min(i + batch_size, len(updates))} rows with genres")
        
        print("Successfully updated all rows with random genres.")
    except sqlite3.Error as e:
        print(f"Error updating genres: {e}")
    finally:
        conn.close()

def verify_updates():
    """Verify that genres were properly added"""
    conn = sqlite3.connect('/home/domas/Desktop/DAT640_MusicProject/DAT640-MusicCRS/databases/songs1.db')
    cursor = conn.cursor()
    
    try:
        # Check total rows
        cursor.execute('SELECT COUNT(*) FROM songs')
        total_rows = cursor.fetchone()[0]
        
        # Check rows with genres
        cursor.execute('SELECT COUNT(*) FROM songs WHERE genre IS NOT NULL')
        rows_with_genres = cursor.fetchone()[0]
        
        # Sample some rows
        cursor.execute('SELECT song_title, artist, genre FROM songs LIMIT 5')
        sample_rows = cursor.fetchall()
        
        print(f"\nVerification Results:")
        print(f"Total rows: {total_rows}")
        print(f"Rows with genres: {rows_with_genres}")
        print("\nSample rows:")
        for row in sample_rows:
            print(f"Title: {row[0]}, Artist: {row[1]}, Genre: {row[2]}")
            
    except sqlite3.Error as e:
        print(f"Error verifying updates: {e}")
    finally:
        conn.close()

# Execute the modifications
if __name__ == "__main__":
    print("Starting database modification...")
    modify_database_schema()
    print("\nUpdating genres...")
    update_genres_in_bulk()
    print("\nVerifying updates...")
    verify_updates()