import streamlit as st
import pandas as pd
from databases.database import *
from prompt import get_chat_completion1, get_chat_completion2
from rag_models import get_rag_model
import datetime
import os
import json

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
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'completed_tasks' not in st.session_state:
    st.session_state.completed_tasks = set()

# Initialize the RAG model
rag_model = get_rag_model()

# Define hardcoded credentials
USERS = {
    "user1": "password1",
    "user2": "password2",
    "admin": "adminpass"
}

# Define key terms/concepts that should be in correct answers - move this outside the function
key_concepts = {
    "task1": {
        "required": ["PSL", "LF11", "ice particle", "rollback", "wet bulb temperature", "LPC"],
        "min_required": 3,
        "feedback_correct": "You've correctly identified the key factors in engine rollback events.",
        "concept_hints": {
            "PSL": "Consider including information about the Propulsion Systems Laboratory (PSL) test facility data.",
            "LF11": "The LF11 engine model (a specific turbofan engine type) is relevant to this analysis.",
            "ice particle": "What size and type of ice particles were critical in the tests?",
            "rollback": "Describe the conditions that lead to engine rollback events (when engine thrust decreases unexpectedly).",
            "wet bulb temperature": "Temperature conditions, particularly wet bulb temperature (a measure that accounts for humidity), play a key role.",
            "LPC": "The Low Pressure Compressor (LPC) region is important to mention in your analysis."
        }
    },
    "task2": {
        "required": ["CALIPSO", "thermal", "SHM", "DAQ", "standby mode", "safe mode", "heater"],
        "min_required": 3,
        "feedback_correct": "Great analysis of the CALIPSO thermal system performance!",
        "concept_hints": {
            "CALIPSO": "Your answer should specifically address the Cloud-Aerosol Lidar and Infrared Pathfinder Satellite Observation (CALIPSO) payload.",
            "thermal": "Focus on the thermal aspects (temperature control and heat management) of the system performance.",
            "SHM": "Include information about the System Health Monitoring (SHM) mode and its thermal characteristics.",
            "DAQ": "The Data Acquisition (DAQ) mode has specific thermal characteristics worth discussing.",
            "standby mode": "How does the thermal system perform in standby mode (reduced power operation)?",
            "safe mode": "Consider the thermal conditions during safe mode operations (emergency power conservation).",
            "heater": "Heater performance and temperature regulation is a critical aspect to evaluate."
        }
    },
    "task3": {
        "required": ["noise", "engine power", "approach", "takeoff", "broadband", "flight velocity", "airframe"],
        "min_required": 3,
        "feedback_correct": "Excellent understanding of aircraft noise profiles!",
        "concept_hints": {
            "noise": "Be more specific about the types of noise components and sources in aircraft operation.",
            "engine power": "How do different engine power settings affect the overall noise generation?",
            "approach": "Consider noise characteristics during approach conditions (when aircraft is descending to land).",
            "takeoff": "Takeoff conditions (maximum power) have distinct noise profiles worth mentioning.",
            "broadband": "Broadband noise (noise distributed across many frequencies) components vary with different conditions.",
            "flight velocity": "How does the speed of the aircraft affect the noise profile?",
            "airframe": "Don't forget to consider airframe noise contributions (noise from the aircraft body, not engines)."
        }
    },
    "task4": {
        "required": ["UPS", "power quality", "safety", "personnel", "equipment", "mission", "critical"],
        "min_required": 3,
        "feedback_correct": "Your UPS system recommendations are well-justified!",
        "concept_hints": {
            "UPS": "Be specific about Uninterruptible Power Supply (UPS) capabilities and implementation.",
            "power quality": "How does a UPS system improve power quality (voltage stability, frequency regulation)?",
            "safety": "Consider safety implications for mission operations when power fluctuations occur.",
            "personnel": "How does UPS implementation affect personnel safety during power events?",
            "equipment": "Discuss protection of sensitive equipment from power surges or outages.",
            "mission": "Relate your answer to mission requirements and continuity of operations.",
            "critical": "Identify which systems are most critical for UPS protection in a space mission context."
        }
    },
    "task5": {
        "required": ["circuit analysis", "electro-mechanical", "safety", "design review", "manufacturing", "critical"],
        "min_required": 3,
        "feedback_correct": "Your analysis technique recommendations are thorough and appropriate!",
        "concept_hints": {
            "circuit analysis": "Specify which circuit analysis techniques (such as FMEA, fault tree analysis) are most effective.",
            "electro-mechanical": "Address the electro-mechanical aspects (where electrical and mechanical systems interact) of the system.",
            "safety": "How do these analytical techniques improve system safety and reliability?",
            "design review": "When in the design review process should these analyses be applied for maximum benefit?",
            "manufacturing": "Consider techniques that identify potential issues before manufacturing begins.",
            "critical": "Explain why these techniques are especially important for high-reliability, mission-critical systems."
        }
    }
}

def evaluate_task_answer(task_id, answer):
    """
    Evaluate user's answer for a specific task.
    Returns (is_correct, feedback_message, missing_concepts)
    """
    # Check which concepts are present and which are missing
    task_concepts = key_concepts[task_id]
    found_concepts = []
    missing_concepts = []
    
    for concept in task_concepts["required"]:
        if concept.lower() in answer.lower():
            found_concepts.append(concept)
        else:
            missing_concepts.append(concept)
    
    is_correct = len(found_concepts) >= task_concepts["min_required"]
    
    if is_correct:
        return True, task_concepts["feedback_correct"], []
    else:
        # Return the list of missing concepts for targeted feedback
        return False, task_concepts["feedback_correct"], missing_concepts

# Function to log user answers2
def log_user_answer(username, task_id, answer, is_correct):
    log_dir = "user_answers"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "username": username,
        "task_id": task_id,
        "answer": answer,
        "is_correct": is_correct,
        "timestamp": timestamp
    }
    
    log_file = os.path.join(log_dir, f"{username}_answers.json")
    
    # Load existing logs if file exists
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Add new entry and save
    logs.append(log_entry)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

# Add this function after your other logging functions
def log_user_evaluation(username, evaluation_data):
    """Log user's final evaluation feedback"""
    log_dir = "user_evaluations"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    evaluation_data["timestamp"] = timestamp
    evaluation_data["username"] = username
    
    log_file = os.path.join(log_dir, f"{username}_evaluation.json")
    
    # Save evaluation data
    with open(log_file, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    return True

def process_rag_query(query):
    """Process a query using the RAG model with the selected model"""
    return rag_model.query(query, st.session_state.selected_model)

# Login form
if not st.session_state.logged_in:
    st.title("NASA Lessons Learned Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
else:
    # Add logout button in the sidebar
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
    
    # Display current user
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
    
    # Define tabs here, inside the else block
    tab1, tab2, tab3 = st.tabs(["RAG Query", "Lesson Recommender", "Explore Mission Database"])
    
    with tab3:
        messages = st.container(height=500)
        messages.chat_message("assistant").write(st.session_state.bot_msgs[0])

        # Only show user-bot message pairs if there are any
        if st.session_state.user_msgs:
            for i, (user_msg, bot_msg) in enumerate(zip(st.session_state.user_msgs, st.session_state.bot_msgs[1:])):
                messages.chat_message("user").write(user_msg)
                messages.chat_message("assistant").write(bot_msg)

        # Button container below chat
        button_container = st.container(border=True)

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
            st.header("User Evaluation Tasks", divider="gray")
            
            # Initialize task completion state if not exists
            if 'task_completion' not in st.session_state:
                st.session_state.task_completion = {
                    "task1": {"completed": False, "correct": False, "attempts": 0},
                    "task2": {"completed": False, "correct": False, "attempts": 0},
                    "task3": {"completed": False, "correct": False, "attempts": 0},
                    "task4": {"completed": False, "correct": False, "attempts": 0},
                    "task5": {"completed": False, "correct": False, "attempts": 0}
                }
            
            # Task selection and submission system
            selected_task = st.selectbox(
                "Select a task to work on:",
                ["Task 1: Engine Rollback Investigation", 
                 "Task 2: Satellite Thermal System Evaluation",
                 "Task 3: Aircraft Noise Profile Assessment",
                 "Task 4: Critical Power System Design",
                 "Task 5: Electronics System Safety Review"]
            )
            
            task_id = f"task{selected_task[5:6]}"  # Extract task number
            
            # Display task description based on selection
            if selected_task == "Task 1: Engine Rollback Investigation":
                st.markdown("""
                ### Task 1: Engine Rollback Investigation
                You're an aerospace engineer analyzing engine performance in icing conditions. Your team needs to understand what particle characteristics lead to engine rollback events. Research the Propulsion Systems Laboratory (PSL) test data findings to determine critical ice particle sizes and temperature conditions that contribute to these events.
                
                Expected findings should include:
                - Analysis of Propulsion Systems Laboratory (PSL) data points on the LF11 engine model
                - Critical particle size requirements for engine rollback (when thrust unexpectedly decreases)
                - Relevant wet bulb temperature range in the Low Pressure Compressor (LPC) region
                """)
            elif selected_task == "Task 2: Satellite Thermal System Evaluation":
                st.markdown("""
                ### Task 2: Satellite Thermal System Evaluation
                As a thermal engineer reviewing post-launch performance, you need to assess how well the Cloud-Aerosol Lidar and Infrared Pathfinder Satellite Observation (CALIPSO) payload thermal system functioned across different operational modes. Investigate the thermal performance data to prepare a brief report on system stability and margin conditions.
                
                Expected findings should include:
                - Thermal boundary condition performance during System Health Monitoring (SHM) and Data Acquisition (DAQ) modes
                - System behavior in various standby and safe modes (reduced power and emergency operations)
                - Heater performance and temperature control effectiveness
                """)
            elif selected_task == "Task 3: Aircraft Noise Profile Assessment":
                st.markdown("""
                ### Task 3: Aircraft Noise Profile Assessment
                You're working on noise reduction for a new aircraft design. Your task is to understand how engine power settings affect different noise components. Research how noise profiles vary between approach and takeoff conditions to inform your design recommendations.
                
                Expected findings should include:
                - Inlet broadband component behavior (noise distributed across many frequencies) at low power settings
                - Relationship between flight velocity and airframe noise (noise from the aircraft body)
                - Comparative noise levels at high takeoff power
                """)
            elif selected_task == "Task 4: Critical Power System Design":
                st.markdown("""
                ### Task 4: Critical Power System Design
                You're designing power systems for a new space mission with sensitive equipment. Your project manager wants recommendations on implementing Uninterruptible Power Supply (UPS) systems. Research the benefits and applications of UPS in NASA missions to justify your proposal.
                
                Expected findings should include:
                - Safety benefits for personnel and equipment
                - Critical applications for emergency operations
                - Power quality improvement capabilities (voltage stability, frequency regulation)
                """)
            elif selected_task == "Task 5: Electronics System Safety Review":
                st.markdown("""
                ### Task 5: Electronics System Safety Review
                As a systems safety engineer preparing for Critical Design Review, you need to recommend appropriate analysis techniques for a complex electro-mechanical system. Research analytical methods that can identify potential hidden circuit problems before manufacturing begins.
                
                Expected findings should include:
                - Applicable system types for specialized circuit analysis (such as FMEA or fault tree analysis)
                - Optimal implementation timing in the project lifecycle
                - Benefits for high-criticality systems (systems where failure would be catastrophic)
                """)
            
            # Task answer submission
            st.write("Submit your findings:")
            user_answer = st.text_area("Your answer", height=150, key=f"answer_{task_id}")
            
            if st.button("Submit Answer", key=f"submit_{task_id}"):
                # Evaluate the answer
                is_correct, feedback_message, missing_concepts = evaluate_task_answer(task_id, user_answer)
                
                # Update task completion state
                if task_id not in st.session_state.task_completion:
                    st.session_state.task_completion[task_id] = {"completed": False, "correct": False, "attempts": 0}
                
                st.session_state.task_completion[task_id]["attempts"] += 1
                current_attempts = st.session_state.task_completion[task_id]["attempts"]
                st.session_state.task_completion[task_id]["completed"] = True
                st.session_state.task_completion[task_id]["correct"] = is_correct
                
                # Log the user's answer
                log_user_answer(st.session_state.username, task_id, user_answer, is_correct)
                
                # Show feedback
                if is_correct:
                    st.success(f"✅ Correct! {feedback_message}")
                    st.session_state.completed_tasks.add(task_id)
                else:
                    # Get the task's concept hints
                    concept_hints = key_concepts[task_id]["concept_hints"]
                    
                    # Select 2 missing concepts to provide hints for (or fewer if less are missing)
                    num_hints = min(2, len(missing_concepts))
                    selected_missing = missing_concepts[:num_hints]
                    
                    # Create targeted feedback
                    hint_text = "Consider including these key elements: "
                    for concept in selected_missing:
                        hint_text += f"\n• {concept_hints[concept]}"
                    
                    st.error(f"❌ Not quite right. Your answer needs more detail. {hint_text}")
            
            # Display task status summary
            st.divider()
            st.subheader("Task Progress")
            
            for i in range(1, 6):
                task_key = f"task{i}"
                task_data = st.session_state.task_completion[task_key]
                
                if not task_data["completed"]:
                    status = "⚪ Not attempted"
                    color = "gray"
                elif task_data["correct"]:
                    status = "✅ Completed successfully"
                    color = "green"
                else:
                    status = f"❌ Attempted ({task_data['attempts']})"
                    color = "red"
                
                st.markdown(f"**Task {i}**: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)

            # Check if all tasks are completed successfully
            all_tasks_completed = all(st.session_state.task_completion[f"task{i}"]["correct"] for i in range(1, 6))

            # Show evaluation form if all tasks are completed successfully and evaluation not yet submitted
            if all_tasks_completed and 'evaluation_submitted' not in st.session_state:
                st.session_state.show_evaluation_form = True

            # Display the evaluation form in a modal-like container
            if all_tasks_completed and st.session_state.get('show_evaluation_form', False):
                st.markdown("### 🎉 Congratulations on completing all tasks!")
                
                with st.container():
                    st.markdown("""
                    ## Final Evaluation Form
                    Please take a moment to provide feedback on your experience with the NASA Lessons Learned system.
                    Your input helps us improve the learning experience and will be valuable for research purposes.
                    """)
                    
                    with st.form("evaluation_form"):
                        # Part 1: General System Evaluation
                        st.header("Part 1: General System Evaluation")
                        
                        # System usability questions
                        st.subheader("System Usability")
                        usability_score = st.radio(
                            "How would you rate the overall usability of the system?",
                            options=["★", "★★", "★★★", "★★★★", "★★★★★"],
                            horizontal=True
                        )
                        
                        # Task difficulty questions
                        st.subheader("Task Difficulty")
                        
                        # Create a more structured layout for task difficulty
                        task_difficulty = {}
                        for i in range(1, 6):
                            task_difficulty[f"task{i}"] = st.radio(
                                f"Rate the difficulty of Task {i}:",
                                options=["Too Easy", "Easy", "Just Right", "Challenging", "Too Difficult"],
                                horizontal=True,
                                key=f"difficulty_task{i}"
                            )
                        
                        # Learning experience questions
                        st.subheader("Learning Experience")
                        learning_value = st.radio(
                            "How valuable was this experience for learning about NASA missions?",
                            options=["★", "★★", "★★★", "★★★★", "★★★★★"],
                            horizontal=True
                        )
                        
                        knowledge_gain = st.radio(
                            "How much did your knowledge about NASA lessons learned increase?",
                            options=["★", "★★", "★★★", "★★★★", "★★★★★"],
                            horizontal=True
                        )
                        
                        # Part 2: RAG Chat Agent Evaluation
                        st.header("Part 2: RAG Chat Agent Evaluation")
                        
                        # General RAG system feedback
                        st.subheader("Overall AI Assistant Performance")
                        rag_helpfulness = st.radio(
                            "How helpful was the AI assistant in completing your tasks?",
                            options=["★", "★★", "★★★", "★★★★", "★★★★★"],
                            horizontal=True
                        )
                        
                        # Model-specific evaluations
                        st.subheader("Model-Specific Evaluation")
                        
                        # OpenAI (ChatGPT) evaluation
                        st.markdown("**OpenAI (ChatGPT) Model**")
                        openai_accuracy = st.radio(
                            "How would you rate the accuracy of the OpenAI (ChatGPT) responses?",
                            options=["★", "★★", "★★★", "★★★★", "★★★★★"],
                            horizontal=True,
                            key="openai_accuracy"
                        )
                        
                        openai_relevance = st.radio(
                            "How relevant were the OpenAI (ChatGPT) responses to your queries?",
                            options=["★", "★★", "★★★", "★★★★", "★★★★★"],
                            horizontal=True,
                            key="openai_relevance"
                        )
                        
                        openai_speed = st.radio(
                            "How would you rate the response speed of the OpenAI (ChatGPT) model?",
                            options=["★", "★★", "★★★", "★★★★", "★★★★★"],
                            horizontal=True,
                            key="openai_speed"
                        )
                        
                        # Llama model evaluation
                        st.markdown("**Llama Model**")
                        llama_accuracy = st.radio(
                            "How would you rate the accuracy of the Llama model responses?",
                            options=["★", "★★", "★★★", "★★★★", "★★★★★"],
                            horizontal=True,
                            key="llama_accuracy"
                        )
                        
                        llama_relevance = st.radio(
                            "How relevant were the Llama model responses to your queries?",
                            options=["★", "★★", "★★★", "★★★★", "★★★★★"],
                            horizontal=True,
                            key="llama_relevance"
                        )
                        
                        llama_speed = st.radio(
                            "How would you rate the response speed of the Llama model?",
                            options=["★", "★★", "★★★", "★★★★", "★★★★★"],
                            horizontal=True,
                            key="llama_speed"
                        )
                        
                        # Model comparison
                        st.subheader("Model Comparison")
                        preferred_model = st.radio(
                            "Which model did you prefer overall?",
                            options=["OpenAI (ChatGPT)", "Llama", "No preference"],
                            horizontal=True
                        )
                        
                        model_preference_reason = st.text_area(
                            "Why did you prefer this model?",
                            height=100
                        )
                        
                        # Part 3: Research-Specific Questions
                        st.header("Part 3: Research-Specific Questions")
                        
                        # Information retrieval effectiveness
                        st.subheader("Information Retrieval")
                        
                        retrieval_quality = st.radio(
                            "How would you rate the quality of retrieved information?",
                            options=["★", "★★", "★★★", "★★★★", "★★★★★"],
                            horizontal=True
                        )
                        
                        # Comparison to traditional methods
                        st.subheader("Comparison to Traditional Methods")
                        
                        traditional_comparison = st.radio(
                            "Compared to traditional document search methods, this system is:",
                            options=["Much worse", "Worse", "About the same", "Better", "Much better"],
                            horizontal=True
                        )
                        
                        time_saving = st.radio(
                            "How much time do you think this system saved you compared to manual research?",
                            options=["No time saved", "A little time", "Moderate time", "Significant time", "Extensive time"],
                            horizontal=True
                        )
                        
                        # Open-ended feedback
                        st.header("Additional Feedback")
                        
                        improvement_suggestions = st.text_area(
                            "What suggestions do you have for improving the system?",
                            height=100
                        )
                        
                        favorite_feature = st.text_area(
                            "What was your favorite feature of the system?",
                            height=100
                        )
                        
                        
                        # Submit button
                        submitted = st.form_submit_button("Submit Evaluation")
                        
                        if submitted:
                            # Collect all evaluation data
                            evaluation_data = {
                                # General system evaluation
                                "usability_score": len(usability_score),
                                "task_difficulty": task_difficulty,
                                "learning_value": len(learning_value),
                                "knowledge_gain": len(knowledge_gain),
                                
                                # RAG chat agent evaluation
                                "rag_helpfulness": len(rag_helpfulness),
                                
                                # Model-specific evaluation
                                "openai_accuracy": len(openai_accuracy),
                                "openai_relevance": len(openai_relevance),
                                "openai_speed": len(openai_speed),
                                "llama_accuracy": len(llama_accuracy),
                                "llama_relevance": len(llama_relevance),
                                "llama_speed": len(llama_speed),
                                "preferred_model": preferred_model,
                                "model_preference_reason": model_preference_reason,
                                
                                # Research-specific evaluation
                                "retrieval_quality": len(retrieval_quality),
                                "traditional_comparison": traditional_comparison,
                                "time_saving": time_saving,
                                
                                # Open-ended feedback
                                "improvement_suggestions": improvement_suggestions,
                                "favorite_feature": favorite_feature
                            }
                            
                            # Log the evaluation
                            log_user_evaluation(st.session_state.username, evaluation_data)
                            
                            # Update session state
                            st.session_state.evaluation_submitted = True
                            st.session_state.show_evaluation_form = False
                            
                            # Show success message
                            st.success("Thank you for your feedback! Your evaluation has been submitted successfully.")
                            st.balloons()

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

    with tab1:
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
