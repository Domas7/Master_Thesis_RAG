from openai import OpenAI
from keys import get_api_keys



def get_chat_completion(instructions, specifications):
    priv = get_api_keys()
    client = OpenAI(api_key=priv)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": instructions
            },

            {
                "role": "user",
                "content": specifications
            }
        ]
    )

    print(f"AI completion: {completion.choices[0].message.content}")
    return completion.choices[0].message.content




def get_chat_completion1(userinput):
    instructions = """
    You are an assistant trying to help a user build a playlist by interpreting the users intentions though natural language. 
    These are the available commands for building the playlist:
    /add songname
    /remove songname
    /list
    /clear
    /add-specific song_title;artist;album_title
    /add-many count mood


    These are the available questions you can write:
    When was album X released?
    How many Albums has X released?
    Which album features song X
    Command not found

    These are examples:
    This is the user input: Hello i want to add Tanssi vaan to the playlist
    This is what you should reply: /add Tanssi vaan

    This is the user input: Hello i want to remove Tanssi vaan from the playlist
    This is what you should reply: /remove Tanssi vaan 

    This is the user input: Hello i want to list all songs
    This is what you should reply: /list

    This is the user input: Hello i want to clear the playlist
    This is what you should reply: /clear

    This is the user input: Hi i want to add Love Story by Taylor Swift from the album Love Story
    This is what you should reply: /add-specific Love Story;Taylor Swift;Love Story

    This is the user input: Could you please tell me when the album Love Story was released?
    This is what you shold reply: When was album Love Story released?

    This is the user input: Hi, i want to know how many albums Taylor Swift has released
    This is what you should reply: How many albums has Taylor Swift released?

    This is the user input: In what album is the song Love Story featured?
    This is what you should reply: Which album features song Love Story

    This is the user input: Can you add 3 songs like Sweet Dreams?
    This is what you should reply: /add-many 3 Sweet Dreams

    This is the user input: I want 5 songs similar to Highway to Hell
    This is what you should reply: /add-many 5 Highway to Hell

    This is the user input: Add 2 songs like Taylor Swift Love Story
    This is what you should reply: /add-many 2 Taylor Swift Love Story

    If the userinput does not match any of the intended functions or questions, you should respond with "command not found or no data was found in the database".
    """

    specifications = f"""
    When writing a command the command should not have a space between the / and the comand word. Ie. the command should be written as /add or /list
    This is the user input that you should interpret: {userinput}
    """

    priv = get_api_keys()
    client = OpenAI(api_key=priv)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": instructions
            },

            {
                "role": "user",
                "content": specifications
            }
        ]
    )

    print(f"AI completion: {completion.choices[0].message.content}")

    return completion.choices[0].message.content




def get_chat_completion2(userinput, songlist):
    instructions = """
    You are an assistant trying to help a user build a playlist by interpreting the users intentions though natural language.
    The user will write its input. In addition to the input there will be a list of songs the users is interacting with as context.
    This is an example of what a song list might look like:
    Each object in the list consists of a tuble by (song title, artist, album title, release year)
    [('Today Was A Fairytale', 'Taylor Swift', "Valentine's Day OST", 2010), ('Secret Smile', 'Booty Callers', 'Clubland 15', 2009), ('Forever & Always', 'Taylor Swift', 'Fearless', 2008), ('Sister', 'The Parson Red Heads', 'King Giraffe', 2007), ('Say Goodnight', 'The Parson Red Heads', 'LIVE [Mondays (at the Echo) - Vol. 3 December 18th 2006]', 2007), ('Mayday', 'Axel Coon', 'Mayday', 2007), ('Hey_ Man (Dry Off)', 'The Parson Red Heads', 'King Giraffe', 2007), ('Traveling to Different Planets', 'The Parson Red Heads', 'King Giraffe', 2007), ('Promise Me', 'Axel Coon', 'Promise Me', 2007), ('The Outside', 'Taylor Swift', 'Taylor Swift', 2006)]

    These is the command you can respond with:
    /add-multiple songid1,songid2,songid3

    These are of how userinteractions should be:

    User input: Hello I want to add the first two songs. 
    This is the list of songs: [('Today Was A Fairytale', 'Taylor Swift', "Valentine's Day OST", 2010), ('Secret Smile', 'Booty Callers', 'Clubland 15', 2009), ('Forever & Always', 'Taylor Swift', 'Fearless', 2008), ('Sister', 'The Parson Red Heads', 'King Giraffe', 2007), ('Say Goodnight', 'The Parson Red Heads', 'LIVE [Mondays (at the Echo) - Vol. 3 December 18th 2006]', 2007), ('Mayday', 'Axel Coon', 'Mayday', 2007), ('Hey_ Man (Dry Off)', 'The Parson Red Heads', 'King Giraffe', 2007), ('Traveling to Different Planets', 'The Parson Red Heads', 'King Giraffe', 2007), ('Promise Me', 'Axel Coon', 'Promise Me', 2007), ('The Outside', 'Taylor Swift', 'Taylor Swift', 2006)]
    This is what you should respond with: /add-multiple 0,1

    User input: Hello I want to add all the songs.
    This is the list of songs: [('Today Was A Fairytale', 'Taylor Swift', "Valentine's Day OST", 2010), ('Secret Smile', 'Booty Callers', 'Clubland 15', 2009), ('Forever & Always', 'Taylor Swift', 'Fearless', 2008), ('Sister', 'The Parson Red Heads', 'King Giraffe', 2007), ('Say Goodnight', 'The Parson Red Heads', 'LIVE [Mondays (at the Echo) - Vol. 3 December 18th 2006]', 2007), ('Mayday', 'Axel Coon', 'Mayday', 2007), ('Hey_ Man (Dry Off)', 'The Parson Red Heads', 'King Giraffe', 2007), ('Traveling to Different Planets', 'The Parson Red Heads', 'King Giraffe', 2007), ('Promise Me', 'Axel Coon', 'Promise Me', 2007), ('The Outside', 'Taylor Swift', 'Taylor Swift', 2006)]
    This is what you should respond with: /add-multiple 0,1,2,3,4,5,6,7,8,9

    User input: Hello I want to add song Today was a fairytale and Forever & Always
    This is the list of songs: [('Today Was A Fairytale', 'Taylor Swift', "Valentine's Day OST", 2010), ('Secret Smile', 'Booty Callers', 'Clubland 15', 2009), ('Forever & Always', 'Taylor Swift', 'Fearless', 2008), ('Sister', 'The Parson Red Heads', 'King Giraffe', 2007), ('Say Goodnight', 'The Parson Red Heads', 'LIVE [Mondays (at the Echo) - Vol. 3 December 18th 2006]', 2007), ('Mayday', 'Axel Coon', 'Mayday', 2007), ('Hey_ Man (Dry Off)', 'The Parson Red Heads', 'King Giraffe', 2007), ('Traveling to Different Planets', 'The Parson Red Heads', 'King Giraffe', 2007), ('Promise Me', 'Axel Coon', 'Promise Me', 2007), ('The Outside', 'Taylor Swift', 'Taylor Swift', 2006)]
    This is what you should respond with: /add-multiple 0,
    """

    specifications = f"""
    This is the user input that you should interpret: {userinput}
    This is the songlist: {songlist}
    """

    priv = get_api_keys()
    client = OpenAI(api_key=priv)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": instructions
            },

            {
                "role": "user",
                "content": specifications
            }
        ]
    )

    print(f"AI completion: {completion.choices[0].message.content}")

    return completion.choices[0].message.content

def get_chat_completion3(userinput):
    instructions = """
    You are an assistant trying to help a user build a playlist by interpreting the users intentions though natural language. 
    These are the available commands for building the playlist:

    /add-many count genre_or_mood

    These are examples:
    This is the user input: Can you add 3 sad songs to my playlist?
    This is what you should reply: /add-many 3 sad

    This is the user input: I want 5 rock songs in my playlist
    This is what you should reply: /add-many 5 rock

    This is the user input: Add 2 happy songs please
    This is what you should reply: /add-many 2 happy
    """

    specifications = f"""
    When writing a command the command should not have a space between the / and the command word.
    This is the user input that you should interpret: {userinput}
    """

    priv = get_api_keys()
    client = OpenAI(api_key=priv)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": instructions
            },

            {
                "role": "user",
                "content": specifications
            }
        ]
    )

    print(f"AI completion: {completion.choices[0].message.content}")

    return completion.choices[0].message.content



if __name__ == "__main__":
    # instructions = """
    # You are an assistant trying to interpret user intentions. 
    # A user may add song to playlist by songname, remove song from playlist by song name, list out all the song in the playlist or clear the playlist.
    # These are the available commands you can write:
    # /add songname
    # /remove songname
    # /list
    # /clear
    # /add-specific song_title;artist;album_title
    # When was album X released?
    # How many Albums has X released?
    # Which album features song X
    # Command not found

    # These are examples:
    # This is the user input: Hello i want to add Tanssi vaan to the playlist
    # This is what you should reply: /add Tanssi vaan

    # This is the user input: Hello i want to remove Tanssi vaan from the playlist
    # This is what you should reply: /remove Tanssi vaan 

    # This is the user input: Hello i want to list all songs
    # This is what you should reply: /list

    # This is the user input: Hello i want to clear the playlist
    # This is what you should reply: /clear

    # This is the user input: Hi i want to add Love Story by Taylor Swift from the album Love Story
    # This is what you should reply: /add-specific Love Story;Taylor Swift;Love Story

    # This is the user input: Could you please tell me when the album Love Story was released?
    # This is what you shold reply: When was album Love Story released?

    # This is the user input: Hi, i want to know how many albums Taylor Swift has released
    # This is what you should reply: How many albums has Taylor Swift released?

    # This is the user input: In what album is the song Love Story featured?
    # This is what you should reply: Which album features song Love Story?

    # If the userinput does not match any of the intended functions or questions, you should respond with command not found.
    # """

    # specifications = f"""
    # This is the user input: In which album is Shine Bright featured?
    # """

    # completion = get_chat_completion(instructions, specifications)
    # print(completion)


    inp = "Hello I want to add none of the songs"
    li = """[('Today Was A Fairytale', 'Taylor Swift', "Valentine's Day OST", 2010), ('Secret Smile', 'Booty Callers', 'Clubland 15', 2009), ('Forever & Always', 'Taylor Swift', 'Fearless', 2008), ('Sister', 'The Parson Red Heads', 'King Giraffe', 2007), ('Say Goodnight', 'The Parson Red Heads', 'LIVE [Mondays (at the Echo) - Vol. 3 December 18th 2006]', 2007), ('Mayday', 'Axel Coon', 'Mayday', 2007), ('Hey_ Man (Dry Off)', 'The Parson Red Heads', 'King Giraffe', 2007), ('Traveling to Different Planets', 'The Parson Red Heads', 'King Giraffe', 2007), ('Promise Me', 'Axel Coon', 'Promise Me', 2007), ('The Outside', 'Taylor Swift', 'Taylor Swift', 2006)]"""
    completion = get_chat_completion2(inp, li)
