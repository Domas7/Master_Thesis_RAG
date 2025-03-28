import streamlit as st
from main import *
import pandas as pd
from databases.database import *
from supporting_functions import *
from prompt import get_chat_completion1, get_chat_completion2
from rag_models import get_rag_model

# Initialize session state variables if they don't exist
playlist = get_playlist_song_titles()
if 'user_msgs' not in st.session_state:
    st.session_state.user_msgs = []
if 'bot_msgs' not in st.session_state:
    st.session_state.bot_msgs = [f"Welcome to the NASA Lessons Learned app. You can ask questions about NASA missions and documents, or use the playlist feature to save interesting papers."]
if 'show_song_selection' not in st.session_state:
    st.session_state.show_song_selection = False
if 'song_options' not in st.session_state:
    st.session_state.song_options = []
if "recommended" not in st.session_state:
    st.session_state.recommended = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "openai"  # Default model

# Initialize the RAG model
rag_model = get_rag_model()

def count_playlist_songs():
    songs = get_playlist_song_titles()
    return f"No data was found in the database"

def get_most_frequent_artist():
    conn = sqlite3.connect('databases/minorTable.db')
    cursor = conn.cursor()
    
    # Get all songs in the playlist
    playlist_songs = get_playlist_songs()
    if not playlist_songs:
        return "No data was found in the database"
    
    # Count artist appearances
    artist_count = {}
    for song in playlist_songs:
        artist = song[3]  # antar at artist er på index 3, pass på hvis det blir changes i DB
        artist_count[artist] = artist_count.get(artist, 0) + 1
    
    if not artist_count:
        return "No data was found in the database"
    
    # Find most frequent artist
    most_frequent = max(artist_count.items(), key=lambda x: x[1])
    return f"{most_frequent[0]} appears the most in the playlist with {most_frequent[1]} songs."

def get_average_release_date():
    conn = sqlite3.connect('databases/minorTable.db')
    cursor = conn.cursor()

    playlist_songs = get_playlist_songs()
    if not playlist_songs:
        return "No data was found in the database"
    
    valid_dates = []
    for song in playlist_songs:
        release_date = song[4]  # Assuming the release date is at index 4
        print(f"release_Dates are {release_date}")

        if release_date and release_date != '0':
            try:
                # Convert the byte data to a valid year
                year = convert_bytes_to_year(release_date)
                if year > 0:
                    valid_dates.append(year)
            except (ValueError, TypeError):
                continue
    
    if not valid_dates:
        return "No valid release dates found in the playlist."
    
    avg_year = sum(valid_dates) / len(valid_dates)
    return f"The average release year of songs in the playlist is {int(avg_year)}"

def process_rag_query(query):
    """Process a query using the RAG model with the selected model"""
    return rag_model.query(query, st.session_state.selected_model)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Explore Mission Database", "Lesson Recommender", "RAG Query"])

with tab1:
    messages = st.container(height=500)
    messages.chat_message("assistant").write(st.session_state.bot_msgs[0])

    # Only show user-bot message pairs if there are any
    if st.session_state.user_msgs:
        for i, (user_msg, bot_msg) in enumerate(zip(st.session_state.user_msgs, st.session_state.bot_msgs[1:])):
            messages.chat_message("user").write(user_msg)
            messages.chat_message("assistant").write(bot_msg)

    # Button container below chat
    button_container = st.container(border=True)

    with button_container:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Focus on Approach"):
                response = count_playlist_songs()
                st.session_state.bot_msgs.append(response)
                st.session_state.user_msgs.append("Focusing on Approach")
                st.rerun()
                
            if st.button("Focus on Challenges"):
                response = get_most_frequent_artist()
                st.session_state.bot_msgs.append(response)
                st.session_state.user_msgs.append("Which artist appears the most in the playlist?")
                st.rerun()
                
            if st.button("Focus on Lessons Learned"):
                response = get_average_release_date()
                st.session_state.bot_msgs.append(response)
                st.session_state.user_msgs.append("What is the average release year?")
                st.rerun()
                
        with col2:
            # Input fields and buttons for queries requiring user input
            with st.expander("When was paper released?"):
                album_name = st.text_input("Paper title", key="album_input")
                if st.button("Check Release Date"):
                    if album_name:
                        response = get_album_date(album_name)
                        st.session_state.bot_msgs.append(response)
                        st.session_state.user_msgs.append(f"When was album {album_name} released?")
                        st.rerun()
            
            with st.expander("What papers are related to this author?"):
                artist_name = st.text_input("Author name", key="artist_input")
                if st.button("Check Papers"):
                    if artist_name:
                        response = how_many_albums(artist_name)
                        st.session_state.bot_msgs.append(response)
                        st.session_state.user_msgs.append(f"How many albums has {artist_name} released?")
                        st.rerun()

    if prompt := st.chat_input("Say something"):
        # Processing prompt
        st.session_state.user_msgs.append(prompt)
        # prompt = prompt + f"This is the playlist: {playlist}"
        if prompt[0] != "/":
            prompt = get_chat_completion1(prompt)
        prompt_components = prompt.split(" ")
        command = prompt_components[0]
        song = " ".join(prompt_components[1:])

        # Executing command
        if command == "/add":
            reply = add(song)
            if isinstance(reply, list):
                st.session_state.show_song_selection = True
                st.session_state.song_options = reply
                st.session_state.bot_msgs.append("Multiple songs found. Please select one from the popup.")
            else:
                st.session_state.bot_msgs.append(reply)
        elif command == "/add-specific":
            reply = add_specific(song)
            st.session_state.bot_msgs.append(reply)
        elif command == "/remove":
            reply = remove(song)
            st.session_state.bot_msgs.append(reply)
        elif command == "/clear":
            reply = clear()
            st.session_state.bot_msgs.append(reply)
        elif command == "/list":
            st.session_state.bot_msgs.append(f"Here is the playlist:\n{playlist}")
        elif command == "/add-many":
            try:
                count = int(prompt_components[1])
                genre_or_mood = " ".join(prompt_components[2:])
                reply = add_many(count, genre_or_mood)
                st.session_state.bot_msgs.append(reply)
            except (ValueError, IndexError):
                st.session_state.bot_msgs.append("Invalid format. Please specify count and genre/mood.")
        else:
            if "When was album" in prompt or "when was album" in prompt.lower():
                split = prompt.strip().split("album")
                album_comps = split[1].split(" ")    
                album = " ".join(album_comps[:-1])
                album = album.strip()
                response = get_album_date(album)
                st.session_state.bot_msgs.append(response)
            elif "How many albums" in prompt or "how many albums" in prompt.lower():
                split = prompt.strip().split("has")
                artist_comps = split[1].split(" ")    
                artist = " ".join(artist_comps[:-1])
                artist = artist.strip()
                response = how_many_albums(artist)
                st.session_state.bot_msgs.append(response)
            elif "Which album features song" in prompt or "which album features song" in prompt.lower():
                split = prompt.strip().split("song")
                song = split[1].strip()
                response = song_album_features(song)
                st.session_state.bot_msgs.append(response)
            else:
                st.session_state.bot_msgs.append("Command not found.")
        
        st.rerun()


    if st.session_state.show_song_selection:
        with st.sidebar:
            st.header("Select a Song")
            for i, song_info in enumerate(st.session_state.song_options):
                title, artist, album = song_info
                st.write(f"🎵 **{title}** by {artist} from _{album}_")
                if st.button("Add", key=f"add_song_{i}"):
                    add_result = add_specific(f"{title};{artist};{album}")
                    st.session_state.user_msgs.append(f"Add specific song {title} by {artist} from album {album}")
                    st.session_state.bot_msgs.append(add_result)
                    st.session_state.show_song_selection = False
                    st.session_state.song_options = []
                    st.rerun()
            
            if st.button("Cancel"):
                st.session_state.show_song_selection = False
                st.session_state.song_options = []
                st.rerun()
               
    with st.sidebar:
        header = st.header("Here will the additional information shown with the title, paper date and more.", divider="gray")
        table = st.table(pd.DataFrame(playlist, columns=["Paper Title", "Paper Author(-s)", "Paper Date", "Paper Link"]))
        st.button(label="Clear 🚮", on_click=clear)

with tab2:
    messages = st.container(height=400)
    messages.chat_message("assistant").write(st.session_state.bot_msgs[0])

    # Only show user-bot message pairs if there are any
    if st.session_state.user_msgs:
        for i, (user_msg, bot_msg) in enumerate(zip(st.session_state.user_msgs, st.session_state.bot_msgs[1:])):
            messages.chat_message("user").write(user_msg)
            messages.chat_message("assistant").write(bot_msg)  

    
    # Get current playlist
    current_playlist = get_playlist_song_titles()
    
    if not current_playlist:
        st.warning("Add some papers you your chosen papers table to get similar recommendations")
    else:
        # Get recommendations
        recommendations = get_song_recommendations(current_playlist, 7)
        st.session_state.recommended = recommendations
        
        if recommendations:
            rec_df = pd.DataFrame(
                recommendations,
                columns=["Song Title", "Artist", "Album Title", "Release Year"]
            )
            
            # Display recommendations table
            st.table(rec_df)
        else:
            st.info("No recommendations found. Try adding more songs to your playlist!")


    if prompt := st.chat_input("Say something", key="recommender"):
        st.session_state.user_msgs.append(prompt)

        prompt = get_chat_completion2(prompt, str(st.session_state.recommended))
        prompt_components = prompt.split(" ")
        command = prompt_components[0]
        songids = prompt_components[1].split(",")

        if command == "/add-multiple":
            response = add_multiple(songids, st.session_state.recommended)
            st.session_state.bot_msgs.append(response)
            st.rerun()

        else:
            st.session_state.bot_msgs.append("Command not found.")
            st.rerun()

with tab3:
    st.header("NASA Mission Knowledge Base")
    
    # Model selection
    model_col1, model_col2 = st.columns(2)
    with model_col1:
        st.write("Select AI Model:")
    with model_col2:
        model_options = ["OpenAI", "Llama"]
        selected_index = 0 if st.session_state.selected_model == "openai" else 1
        selected_model = st.selectbox(
            "Model",
            model_options,
            index=selected_index,
            label_visibility="collapsed"
        )
        st.session_state.selected_model = selected_model.lower()
    
    # Chat interface for RAG
    if 'rag_messages' not in st.session_state:
        st.session_state.rag_messages = [
            {"role": "assistant", "content": "I can answer questions about NASA missions and documents. What would you like to know?"}
        ]
    
    # Display chat messages
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    if rag_query := st.chat_input("Ask about NASA missions...", key="rag_input"):
        # Add user message to chat history
        st.session_state.rag_messages.append({"role": "user", "content": rag_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(rag_query)
        
        # Get response from RAG model
        with st.spinner(f"Thinking using {st.session_state.selected_model.upper()}..."):
            response = process_rag_query(rag_query)
        
        # Add assistant response to chat history
        st.session_state.rag_messages.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)