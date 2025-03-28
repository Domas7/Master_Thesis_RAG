# Codespace for logic and backen functions
from databases.database import *
import sqlite3
from rapidfuzz import fuzz

def get_commands():
    return ["/add","/remove","/clear","/list","/add-many"]

def add(song_name):
    similar_song_name = get_song_title_by_similar_name(song_name)
    status = add_song_to_playlist_by_title(similar_song_name)
    if status == True:
        return f"Added {song_name} to the playlist \nTry out our buttons below to find more about the song! :P"
    if status == False:
        return f"I was not able to add that song, could you ask for another one?"
    else:
        # status is already a list of tuples containing (song_title, artist, album)
        return status

def add_specific(song):
    title, artist, album = song.strip().split(";")
    status = add_specific_song_to_playlist(title, artist, album)
    if status:
        return f"Added Song title: {title}, Artist: {artist}, Album title: {album} to the playlist"
    

def add_multiple(songids, songlist): #request looks like this: [1,2,3,4]
    for id in songids:
        specific_song = songlist[int(id)]
        specific_song_string = f"{specific_song[0]};{specific_song[1]};{specific_song[2]}"
        add_specific(specific_song_string)

    return f"Added the recommended songs to the playlist."


def clear():
    """Clear the playlist"""
    clear_playlist()

    return f"Playlist cleared"


def remove(song_name):
    """Remove a song from the playlist"""
    remove_song_from_playlist_by_title(song_name)

    return f"removed {song_name} from the playlist"


def get_album_date(album):
    datetime = get_album_release_year(album)
    return f"{album} was released in: {datetime}"


def how_many_albums(artist):
    count = get_unique_album_count_by_artist(artist)
    albums = get_albums_by_artist(artist)
    return f"{artist} has released {count} albums named {albums}"


def song_album_features(song):
    albums = get_albums_by_song_title(song)
    response = (f"The song {song}, is featured in these albums: {albums}")

    return response

def add_many(count, *search_terms):
    """Add multiple songs that best match the search terms"""
    conn = sqlite3.connect('databases/minorTable.db')
    cursor = conn.cursor()
    
    # Get all songs from database
    cursor.execute('SELECT song_title, artist, album_title FROM songs')
    all_songs = cursor.fetchall()
    conn.close()
    
    # Create search string from all terms
    search_string = " ".join(search_terms).lower()
    
    # Create a list of tuples with (song_info, similarity_score)
    scored_songs = []
    for song in all_songs:
        # Combine song title and artist for matching
        song_string = f"{song[0]} {song[1]}".lower()
        score = fuzz.ratio(search_string, song_string)
        scored_songs.append((song, score))
    
    # Sort by similarity score and get the top 'count' matches
    scored_songs.sort(key=lambda x: x[1], reverse=True)
    best_matches = scored_songs[:int(count)]
    
    if not best_matches:
        return f"No songs found matching your criteria: {search_string}"
    
    # Add the songs to the playlist
    for song, score in best_matches:
        add_specific(f"{song[0]};{song[1]};{song[2]}")
    
    return f"Added {len(best_matches)} songs similar to '{search_string}' to the playlist."


if __name__ == "__main__":
    playlist = []

    add_song = add("Shine Bright", playlist)
    print(list)