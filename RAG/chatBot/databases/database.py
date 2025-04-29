import sqlite3

######################### SETUP FUNCTIONS ############################################################

def setup_song_database():
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('databases/songs1.db')
    cursor = conn.cursor()
    
    # Create the table with the specified schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS songs (
            song_id INTEGER PRIMARY KEY AUTOINCREMENT,
            song_title TEXT,
            album_title TEXT,
            artist TEXT,
            release_date TEXT
        )
    ''')
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def setup_playlist_database():
    """
    Sets up a playlist database that contains only song_ids.
    """
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('databases/lessons.db')
    conn.execute('PRAGMA foreign_keys = ON;')  # Enable foreign key constraint
    cursor = conn.cursor()
    
    # Create the playlist table with only song_id as a column
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS playlist (
            song_id INTEGER,
            FOREIGN KEY (song_id) REFERENCES songs(song_id)
        )
    ''')
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

################################# INSERT INTO DATABASE FUNCTIONS #########################################################

def insert_song(song_title, album_title, artist, release_date):
    """
    Inserts a song into the songs database.

    Parameters:
        song_title (str): The title of the song.
        album_title (str): The title of the album.
        artist (str): The artist of the song.
        release_date (str): The release date of the song.
        genre (str): The genre of the song.

    Returns:
        int: The song_id of the inserted song.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect('databases/minotTable.db')
    cursor = conn.cursor()
    
    # Insert the song into the songs table
    try:
        cursor.execute('''
            INSERT INTO songs (song_title, album_title, artist, release_date)
            VALUES (?, ?, ?, ?)
        ''', (song_title, album_title, artist, release_date))
        conn.commit()
        song_id = cursor.lastrowid  # Get the auto-generated song_id
    except sqlite3.IntegrityError as e:
        print(f"Error inserting song: {e}")
        song_id = None
    finally:
        conn.close()
    return song_id

def add_song_to_playlist(song_id):
    """
    Adds a song to the playlist by song_id.
    
    Parameters:
        song_id (int): The ID of the song to add to the playlist.
    """
    conn = sqlite3.connect('databases/lessons.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO playlist (song_id)
            VALUES (?)
        ''', (song_id,))
        conn.commit()
    except sqlite3.IntegrityError as e:
        print(f"Error adding song to playlist: {e}")
    finally:
        conn.close()



def add_song_to_playlist_by_title(song_title):
    """
    Adds a song to the playlist by song_title.

    Parameters:
        song_title (str): The title of the song to add to the playlist.
    """
    conn = sqlite3.connect('databases/minotTable.db')
    cursor = conn.cursor()
    cursor.execute('SELECT song_id, song_title, artist, album_title FROM songs WHERE song_title = ?', (song_title,))
    results = cursor.fetchall()
    conn.close()

    if not results:
        print(f"No song found with title '{song_title}'.")
        return False

    elif len(results) == 1:
        # Only one song matches the title
        song_id = results[0][0]
        add_song_to_playlist(song_id)
        print(f"Added '{song_title}' to the playlist.")
        return True
    else:
        # Multiple songs match the title
        return [(result[1], result[2], result[3]) for result in results]
    


def add_specific_song_to_playlist(song_title, artist, album_title):
    """
    Adds a specific song to the playlist by matching song_title, artist, and album_title.

    Parameters:
        song_title (str): The title of the song.
        artist (str): The artist of the song.
        album_title (str): The title of the album.

    Returns:
        bool: True if the song is successfully added, False otherwise.
    """
    # Connect to the songs database
    conn = sqlite3.connect('databases/minotTable.db')
    cursor = conn.cursor()

    # Query for the song
    cursor.execute('''
        SELECT song_id FROM songs WHERE song_title = ? AND artist = ? AND album_title = ?
    ''', (song_title, artist, album_title))
    results = cursor.fetchall()
    conn.close()

    if not results:
        print(f"No song found with title '{song_title}', artist '{artist}', and album '{album_title}'.")
        return False
    elif len(results) == 1:
        song_id = results[0][0]
        # Add to playlist
        add_song_to_playlist(song_id)
        print(f"Added '{song_title}' by '{artist}' from album '{album_title}' to the playlist.")
        return True
    else:
        # Multiple matches found
        print("Multiple songs found with the given details:")
        for idx, (song_id,) in enumerate(results, start=1):
            print(f"{idx}. song_id: {song_id}")
        print("Cannot add song to playlist due to ambiguity.")
        return False



################################### GET RECORDS FROM DATABASES FUNCTIONS ####################################################

def get_songs_by_title(song_title):
    """
    Retrieves songs from the database that match the given song title.

    Parameters:
        song_title (str): The title of the song to search for.

    Returns:
        list of tuples: A list of records matching the song title.
    """
    conn = sqlite3.connect('databases/minotTable.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM songs WHERE song_title = ?
    ''', (song_title,))
    results = cursor.fetchall()

    conn.close()
    return results

def get_songs_by_album(album_title):
    """
    Retrieves songs from the database that are in the given album.

    Parameters:
        album_title (str): The title of the album to search for.

    Returns:
        list of tuples: A list of records matching the album title.
    """
    conn = sqlite3.connect('databases/minotTable.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM songs WHERE album_title = ?
    ''', (album_title,))
    results = cursor.fetchall()

    conn.close()
    return results

def get_songs_by_artist(artist):
    """
    Retrieves songs from the database that match the given artist.

    Parameters:
        artist (str): The artist to search for.

    Returns:
        list of tuples: A list of records matching the artist.
    """
    conn = sqlite3.connect('databases/minotTable.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM songs WHERE artist = ?
    ''', (artist,))
    results = cursor.fetchall()

    conn.close()
    return results

def get_playlist_songs():
    """
    Retrieves all songs from the playlist.

    Returns:
        list of tuples: A list of song records from the playlist.
    """
    conn_playlist = sqlite3.connect('databases/lessons.db')
    cursor_playlist = conn_playlist.cursor()
    
    cursor_playlist.execute('SELECT song_id FROM playlist')
    song_ids = cursor_playlist.fetchall()
    conn_playlist.close()
    
    # Connect to songs database to get song details
    conn_songs = sqlite3.connect('databases/minotTable.db')
    cursor_songs = conn_songs.cursor()
    
    songs = []
    for (song_id,) in song_ids:
        cursor_songs.execute('SELECT * FROM songs WHERE song_id = ?', (song_id,))
        song = cursor_songs.fetchone()
        if song:
            songs.append(song)
    conn_songs.close()
    
    return songs

def get_playlist_song_titles():
    """
    Retrieves all song titles from the playlist.

    Returns:
        list: A list of song titles from the playlist.
    """
    conn_playlist = sqlite3.connect('databases/minorTable.db')
    cursor_playlist = conn_playlist.cursor()
    cursor_playlist.execute('SELECT song_id FROM playlist')
    song_ids = [row[0] for row in cursor_playlist.fetchall()]
    conn_playlist.close()

    conn_songs = sqlite3.connect('databases/minotTable.db')
    cursor_songs = conn_songs.cursor()
    song_info = []
    for song_id in song_ids:
        cursor_songs.execute('SELECT song_title, artist, album_title, release_date FROM songs WHERE song_id = ?', (song_id,))
        result = cursor_songs.fetchone()
        if result:
            song_title, artist, album_title, release_date = result
            release_year = convert_bytes_to_year(release_date)
            song_info.append((song_title, artist, album_title, release_year))    
    conn_songs.close()

    return song_info


def get_album_release_year(album_title):
    """
    Retrieves the release year of the first album found in the database matching the given album title.

    Parameters:
        album_title (str): The title of the album.

    Returns:
        str: The release year of the album, or None if not found.
    """
    # Connect to the songs database
    conn = sqlite3.connect('databases/minotTable.db')
    cursor = conn.cursor()

    # Query for the release date of the album
    cursor.execute('''
        SELECT release_date FROM songs WHERE album_title = ? LIMIT 1
    ''', (album_title,))
    result = cursor.fetchone()
    conn.close()

    print(result)

    if result:
        release_date = convert_bytes_to_datetime(result)
        return release_date

    else:
        print(f"No album found with title '{album_title}'.")
        return None



def get_unique_album_count_by_artist(artist_name):
    """
    Returns the number of unique album titles by the given artist.

    Parameters:
    - artist_name (str): The name of the artist.

    Returns:
    - int: Number of unique albums by the artist.
    """
    db_path = 'databases/minotTable.db'  # Corrected the database path

    # Establish a connection to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # SQL query to count unique album titles by the artist
    query = '''
    SELECT COUNT(DISTINCT album_title)
    FROM songs
    WHERE artist = ?
    '''

    try:
        # Execute the query with the artist's name as a parameter to prevent SQL injection
        cursor.execute(query, (artist_name,))
        result = cursor.fetchone()

        # If the artist is found, return the count; otherwise, return 0
        album_count = result[0] if result[0] is not None else 0
        return album_count

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return 0  # Return 0 in case of error for simplicity

    finally:
        # Close the database connection
        conn.close()



def get_albums_by_song_title(song_title):
    """
    Retrieves a list of all albums that the given song is featured in.

    Parameters:
        song_title (str): The title of the song.

    Returns:
        list: A list of album titles that the song is featured in.
    """

    # Connect to the songs database
    conn = sqlite3.connect('databases/minotTable.db')
    cursor = conn.cursor()

    # Query to find all unique albums that contain the song
    cursor.execute('''
        SELECT DISTINCT album_title FROM songs WHERE song_title = ?
    ''', (song_title,))
    results = cursor.fetchall()

    # Close the database connection
    conn.close()

    # Extract album titles from the query results
    albums = [row[0] for row in results]

    return albums

def get_albums_by_artist(artist):
    """
    Retrieves a list of all albums that the given song is featured in.

    Parameters:
        song_title (str): The title of the song.

    Returns:
        list: A list of album titles that the song is featured in.
    """

    # Connect to the songs database
    conn = sqlite3.connect('databases/minotTable.db')
    cursor = conn.cursor()

    # Query to find all unique albums that contain the song
    cursor.execute('''
        SELECT DISTINCT album_title FROM songs WHERE artist = ?
    ''', (artist,))
    results = cursor.fetchall()

    # Close the database connection
    conn.close()

    # Extract album titles from the query results
    albums = [row[0] for row in results]

    return albums






###################################################### REMOVING SONGS ################################################


def remove_song_from_playlist_by_title(song_title):
    """
    Removes a song from the playlist by song_title.

    Parameters:
        song_title (str): The title of the song to remove from the playlist.
    """
    # Connect to the songs database to find the song_id(s)
    conn = sqlite3.connect('databases/minotTable.db')
    cursor = conn.cursor()
    cursor.execute('SELECT song_id, artist, album_title FROM songs WHERE song_title = ?', (song_title,))
    results = cursor.fetchall()
    conn.close()

    if not results:
        print(f"No song found with title '{song_title}'.")
        return

    # Connect to the playlist database to remove the song
    conn_playlist = sqlite3.connect('databases/lessons.db')
    cursor_playlist = conn_playlist.cursor()

    # Find all matching songs in the playlist
    playlist_song_ids = []
    for song_id, artist, album_title in results:
        cursor_playlist.execute('SELECT song_id FROM playlist WHERE song_id = ?', (song_id,))
        if cursor_playlist.fetchone():
            playlist_song_ids.append((song_id, artist, album_title))

    if not playlist_song_ids:
        print(f"No songs with title '{song_title}' are currently in the playlist.")
        conn_playlist.close()
        return

    for song in playlist_song_ids:
        song_id = song[0]
        cursor_playlist.execute('DELETE FROM playlist WHERE song_id = ?', (song_id,))
        conn_playlist.commit()

    conn_playlist.close()
    return



def clear_playlist():
    """
    Removes all songs from the playlist.
    """
    # Connect to the playlist database
    conn = sqlite3.connect('databases/lessons.db')
    cursor = conn.cursor()
    
    # Delete all entries from the playlist table
    cursor.execute('DELETE FROM playlist')
    conn.commit()
    
    # Close the connection
    conn.close()
    print("All songs have been removed from the playlist.")



def get_song_recommendations(playlist_songs, limit=5):
    """
    Get song recommendations based on the current playlist.
    
    Args:
        playlist_songs: List of songs currently in the playlist (title, artist, album, year)
        limit: Number of recommendations to return
    
    Returns:
        List of recommended songs as tuples (title, artist, album, year)
    """
    if not playlist_songs:
        return []
        
    conn = sqlite3.connect('databases/minotTable.db')
    cursor = conn.cursor()
    
    # Get current playlist song titles for exclusion
    playlist_titles = [song[0] for song in playlist_songs]
    playlist_titles_str = ','.join('?' * len(playlist_titles))
    
    recommendations = set()
    
    for song in playlist_songs:
        title, artist, album, year = song
        
        # Find songs by the same artist
        cursor.execute(f'''
            SELECT DISTINCT song_title, artist, album_title, release_date 
            FROM songs 
            WHERE artist = ? 
            AND song_title NOT IN ({playlist_titles_str})
            LIMIT 5
        ''', (artist, *playlist_titles))
        recommendations.update(cursor.fetchall())
        
        # Find songs from the same year (±2 years)
        if year:
            try:
                year = int(year)
                cursor.execute(f'''
                    SELECT DISTINCT song_title, artist, album_title, release_date 
                    FROM songs 
                    WHERE CAST(release_date AS INTEGER) BETWEEN ? AND ?
                    AND song_title NOT IN ({playlist_titles_str})
                    AND artist != ?
                    LIMIT 5
                ''', (year - 2, year + 2, *playlist_titles, artist))
                recommendations.update(cursor.fetchall())
            except (ValueError, TypeError):
                continue
        
        # Find songs from the same album
        cursor.execute(f'''
            SELECT DISTINCT song_title, artist, album_title, release_date 
            FROM songs 
            WHERE album_title = ? 
            AND song_title NOT IN ({playlist_titles_str})
            LIMIT 3
        ''', (album, *playlist_titles))
        recommendations.update(cursor.fetchall())
    
    conn.close()
    
    # Convert recommendations to list, process release dates, and limit results
    processed_recommendations = []
    for rec in recommendations:
        title, artist, album, release_date = rec
        year = convert_bytes_to_year(release_date)
        processed_recommendations.append((title, artist, album, year))
    
    # Sort by release year (most recent first) and limit results
    processed_recommendations.sort(key=lambda x: x[3] if x[3] is not None else 0, reverse=True)
    print("processed_recommendations:", str(processed_recommendations[:limit]))
    return processed_recommendations[:limit]



if __name__ == "__main__":
    # # # ################# SETTING UP THE DATABASES
    setup_song_database()
    # # setup_playlist_database()

    # ################ INSERT SONGS INTO SONG DATABASE
    # #song_id1 = insert_song("Hey Jude", "Hey Jude", "The Beatles", "1968-08-26", "Rock")
    # #song_id2 = insert_song("Bohemian Rhapsody", "A Night at the Opera", "Queen", "1975-10-31", "Rock")
    # song3 = insert_song("Beat it", "Beat", "Michael Jackson", "2003", "Pop/Rock")

    # # ############## INSERT SONGS INTO PLAYLIST DATABASE
    # # add_song_to_playlist(1)
    # # add_song_to_playlist(2)

    # # ############## GET PLAYLIST SONG TITLES
    # # playlist_titles = get_playlist_song_titles()
    # # print("Playlist Songs:")
    # # for title in playlist_titles:
    # #     print(title)
