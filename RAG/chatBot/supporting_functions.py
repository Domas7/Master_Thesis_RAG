import struct
from datetime import datetime
import sqlite3
from rapidfuzz import process, fuzz


def convert_bytes_to_datetime(byte_tuple):
    # Unpack the bytes as a little-endian unsigned 64-bit integer
    year = struct.unpack('<Q', byte_tuple[0])[0]

    # Create a datetime object using the extracted year
    date = datetime(year=year, month=1, day=1)
    return date

def convert_bytes_to_year(byte_data):
    # Unpack the first two bytes as a little-endian unsigned short (year)
    year = struct.unpack('<H', byte_data[:2])[0]
    return year



def get_close_matches111(word, possibilities, n=3, cutoff=0.6):
    """
    Use RapidFuzz's process.extract function to return a list of the best "good enough" matches.

    word is a sequence for which close matches are desired (typically a string).

    possibilities is a list of sequences against which to match word (typically a list of strings).

    Optional arg n (default 3) is the maximum number of close matches to return. n must be > 0.

    Optional arg cutoff (default 0.6) is a float in [0, 1]. Possibilities that don't score at least that similar to word are ignored.

    The best (no more than n) matches among the possibilities are returned in a list, sorted by similarity score, most similar first.

    >>> get_close_matches("appel", ["ape", "apple", "peach", "puppy"])
    ['apple', 'ape']
    >>> import keyword as _keyword
    >>> get_close_matches("wheel", _keyword.kwlist)
    ['while']
    >>> get_close_matches("Apple", _keyword.kwlist)
    []
    >>> get_close_matches("accept", _keyword.kwlist)
    ['except']
    """
    if not n > 0:
        raise ValueError(f"n must be > 0: {n}")
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError(f"cutoff must be in [0.0, 1.0]: {cutoff}")

    # Adjust cutoff since RapidFuzz scores are between 0 and 100
    adjusted_cutoff = cutoff * 100
    results = process.extract(
        word, possibilities, scorer=fuzz.ratio, limit=n, score_cutoff=adjusted_cutoff
    )

    return [match for match, score, _ in results]




# Gets companyname based on similar input companyname from app.db
def get_song_title_by_similar_name(songname):
    """
    Retrieves company name based on similar input company name from the app.db database.

    :param songname: The name of the song to retrieve data for.
    :return: A string containing the song name found in the database, or None if the song is not found.
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('databases/minorTable.db')
        c = conn.cursor()
        # Execute a SELECT query to find the song titles
        c.execute("SELECT song_title FROM songs")
        rows = c.fetchall()
        
        # Extract song names from fetched rows, filtering out None values
        song_names = [row[0] for row in rows if row[0] is not None]

        # Check if the exact song name is present in the database
        if songname in song_names:
            return songname
        else:
            # Find the most similar song name
            similar_names = get_close_matches111(songname, song_names, n=1, cutoff=0.70)
            if similar_names:
                return similar_names[0]  # Return the first (most similar) name from the list
            else:
                print("Song not found.")
                return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        if conn:
            conn.close()




if __name__ == "__main__":
    name = get_song_title_by_similar_name("Tansssi vaan")
    print(name)


