import streamlit as st
import pandas as pd
from rag_models import get_rag_model
import datetime
import os
import json
import random

# Initialize session state variables if they don't exist
if 'user_msgs' not in st.session_state:
    st.session_state.user_msgs = []
if 'bot_msgs' not in st.session_state:
    st.session_state.bot_msgs = [f"Welcome to the Master Thesis Application that focuses on supporting human decision making using Retrieval Augmented Generation (RAG) - a technology that combines AI with relevant information from trusted documents to provide more accurate and factual responses.\n\nAs part of the thesis evaluation, you'll complete 5 engineering tasks displayed in the sidebar. You'll take on the role of different engineers solving real-world problems. If a task proves particularly challenging, you may skip it after 4 attempts. Once you've completed or skipped all tasks, a feedback form will appear to help improve the application. Thank you for your participation!"]
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
# Initialize task model assignments if not exists
if 'task_models' not in st.session_state:
    # Randomly assign models to tasks
    st.session_state.task_models = {
        f"task{i}": random.choice(["openai", "llama"]) for i in range(1, 6)
    }
if 'can_select_model' not in st.session_state:
    st.session_state.can_select_model = False
if 'current_task_id' not in st.session_state:
    st.session_state.current_task_id = None
if 'show_feedback_popup' not in st.session_state:
    st.session_state.show_feedback_popup = False
if 'skipped_tasks' not in st.session_state:
    st.session_state.skipped_tasks = set()

# Initialize the RAG model
rag_model = get_rag_model()

# Define hardcoded credentials
USERS = {
    "user1": "password1",
    "user2": "password2",
    "admin": "adminpass",
    "bruker1": "passord1",
    "bruker2": "passord2",
    "bruker3": "passord3",
    "bruker4": "passord4",
    "Snorre": "Snorre123",
    "Martin": "Martin123",
    "Christoffer": "Christoffer123",
    "Marius": "Marius123",
    "Edvard": "Edvard123",
    "Daniel": "Daniel123",
    "Stine": "Stine123",
    "Eirik": "Eirik123",
    "Fredrik": "Fredrik123",
    "Emerson": "Emerson123",
    "Johan": "Johan123",
    "Dominykas": "Dominykas123",
    "Chiran": "Chiran123",
    "Filip": "Filip123",
    "Sina": "Sina123",
    "Håvard": "Håvard123",
    "Kevin": "Kevin123",
    "Jonathan": "Jonathan123",
    "Tord": "Tord123",
    "Patrik": "Patrik123",
    "Kien": "Kien123",
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

# Function to log user answers with model information
def log_user_answer(username, task_id, answer, is_correct, model_used=None, query=None):
    log_dir = "user_answers"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "username": username,
        "task_id": task_id,
        "answer": answer,
        "is_correct": is_correct,
        "timestamp": timestamp,
        "entry_type": "query" if task_id.startswith("query") or is_correct == "not_submitted" else "submission"
    }
    
    # Add model and query information if provided
    if model_used:
        log_entry["model_used"] = model_used
    if query:
        log_entry["query"] = query
    
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

def check_all_tasks_completed_or_skipped():
    """Check if all tasks are either completed or skipped"""
    for i in range(1, 6):
        task_key = f"task{i}"
        if (not st.session_state.task_completion[task_key]["correct"] and 
            task_key not in st.session_state.skipped_tasks):
            return False
    return True

def process_rag_query(query, task_id=None):
    """Process a query using the RAG model with the selected model"""
    # If task_id is provided, use it; otherwise use the current active task
    task_to_use = task_id if task_id else st.session_state.current_task_id
    
    # If a task is active and model selection is disabled, use the assigned model for that task
    if task_to_use and not st.session_state.can_select_model:
        model_to_use = st.session_state.task_models[task_to_use]
    else:
        model_to_use = st.session_state.selected_model
    
    result = rag_model.query(query, model_to_use)
    
    # Log the query and model used
    if st.session_state.logged_in:
        log_user_answer(
            st.session_state.username, 
            "query" if task_to_use is None else f"query_{task_to_use}", 
            result, 
            "not_submitted",  # Mark as not submitted since this is just a query
            model_used=model_to_use,
            query=query
        )
    
    return result

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
    tab1, tab3 = st.tabs(["RAG Query", "About"])
    
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
            # Store the currently selected task in session state
            st.session_state.current_task_id = task_id
            
            # Display task description based on selection
            if selected_task == "Task 1: Engine Rollback Investigation":
                st.markdown("""
                ### Task 1: Engine Rollback Investigation
                """)
                # Make the task description non-copyable using HTML/CSS
                st.markdown("""
                <div style="user-select: none; -webkit-user-select: none; -ms-user-select: none;">
                You're an aerospace engineer analyzing engine performance in icing conditions. Your team needs to understand what particle characteristics lead to engine rollback events. Research the Propulsion Systems Laboratory (PSL) test data findings to determine critical ice particle sizes and temperature conditions that contribute to these events.
                
                Expected findings should include:
                - Analysis of Propulsion Systems Laboratory (PSL) data points on the LF11 engine model
                - Critical particle size requirements for engine rollback (when thrust unexpectedly decreases)
                - Relevant wet bulb temperature range in the Low Pressure Compressor (LPC) region
                </div>
                """, unsafe_allow_html=True)
            elif selected_task == "Task 2: Satellite Thermal System Evaluation":
                st.markdown("""
                ### Task 2: Satellite Thermal System Evaluation
                """)
                # Make the task description non-copyable
                st.markdown("""
                <div style="user-select: none; -webkit-user-select: none; -ms-user-select: none;">
                As a thermal engineer reviewing post-launch performance, you need to assess how well the Cloud-Aerosol Lidar and Infrared Pathfinder Satellite Observation (CALIPSO) payload thermal system functioned across different operational modes. Investigate the thermal performance data to prepare a brief report on system stability and margin conditions.
                
                Expected findings should include:
                - Thermal boundary condition performance during System Health Monitoring (SHM) and Data Acquisition (DAQ) modes
                - System behavior in various standby and safe modes (reduced power and emergency operations)
                - Heater performance and temperature control effectiveness
                </div>
                """, unsafe_allow_html=True)
            elif selected_task == "Task 3: Aircraft Noise Profile Assessment":
                st.markdown("""
                ### Task 3: Aircraft Noise Profile Assessment
                """)
                # Make the task description non-copyable
                st.markdown("""
                <div style="user-select: none; -webkit-user-select: none; -ms-user-select: none;">
                You're working on noise reduction for a new aircraft design. Your task is to understand how engine power settings affect different noise components. Research how noise profiles vary between approach and takeoff conditions to inform your design recommendations.
                
                Expected findings should include:
                - Inlet broadband component behavior (noise distributed across many frequencies) at low power settings
                - Relationship between flight velocity and airframe noise (noise from the aircraft body)
                - Comparative noise levels at high takeoff power
                </div>
                """, unsafe_allow_html=True)
            elif selected_task == "Task 4: Critical Power System Design":
                st.markdown("""
                ### Task 4: Critical Power System Design
                """)
                # Make the task description non-copyable
                st.markdown("""
                <div style="user-select: none; -webkit-user-select: none; -ms-user-select: none;">
                You're designing power systems for a new space mission with sensitive equipment. Your project manager wants recommendations on implementing Uninterruptible Power Supply (UPS) systems. Research the benefits and applications of UPS in NASA missions to justify your proposal.
                
                Expected findings should include:
                - Safety benefits for personnel and equipment
                - Critical applications for emergency operations
                - Power quality improvement capabilities (voltage stability, frequency regulation)
                </div>
                """, unsafe_allow_html=True)
            elif selected_task == "Task 5: Electronics System Safety Review":
                st.markdown("""
                ### Task 5: Electronics System Safety Review
                """)
                # Make the task description non-copyable
                st.markdown("""
                <div style="user-select: none; -webkit-user-select: none; -ms-user-select: none;">
                As a systems safety engineer preparing for Critical Design Review, you need to recommend appropriate analysis techniques for a complex electro-mechanical system. Research analytical methods that can identify potential hidden circuit problems before manufacturing begins.
                
                Expected findings should include:
                - Applicable system types for specialized circuit analysis (such as FMEA or fault tree analysis)
                - Optimal implementation timing in the project lifecycle
                - Benefits for high-criticality systems (systems where failure would be catastrophic)
                </div>
                """, unsafe_allow_html=True)
            
            # Task answer submission
            st.write("Submit your findings:")
            user_answer = st.text_area("Your answer", height=150, key=f"answer_{task_id}")
            
            # Create columns for Submit and Skip buttons
            col1, col2 = st.columns(2)
            
            with col1:
                submit_button = st.button("Submit Answer", key=f"submit_{task_id}")
            
            with col2:
                # Only show skip button if there have been at least 4 unsuccessful attempts
                can_skip = task_id in st.session_state.task_completion and st.session_state.task_completion[task_id]["attempts"] >= 4
                
                if can_skip:
                    skip_button = st.button("Skip Task", key=f"skip_{task_id}")
                else:
                    # Show disabled skip button with attempts counter
                    attempts = 0
                    if task_id in st.session_state.task_completion:
                        attempts = st.session_state.task_completion[task_id]["attempts"]
                    
                    remaining = max(0, 4 - attempts)
                    st.button(
                        f"Skip ({remaining} more attempts)",
                        key=f"skip_disabled_{task_id}",
                        disabled=True
                    )
                    skip_button = False
            
            if submit_button:
                # Evaluate the answer
                is_correct, feedback_message, missing_concepts = evaluate_task_answer(task_id, user_answer)
                
                # Update task completion state
                if task_id not in st.session_state.task_completion:
                    st.session_state.task_completion[task_id] = {"completed": False, "correct": False, "attempts": 0}
                
                st.session_state.task_completion[task_id]["attempts"] += 1
                current_attempts = st.session_state.task_completion[task_id]["attempts"]
                st.session_state.task_completion[task_id]["completed"] = True
                st.session_state.task_completion[task_id]["correct"] = is_correct
                
                # Log the user's answer with the model used for this task
                log_user_answer(
                    st.session_state.username, 
                    task_id, 
                    user_answer, 
                    is_correct,
                    model_used=st.session_state.task_models[task_id],
                    query=f"Task submission - {task_id}"  # Mark as an actual task submission
                )
                
                # Show feedback
                if is_correct:
                    st.success(f"✅ Correct! {feedback_message}")
                    st.session_state.completed_tasks.add(task_id)
                    
                    # Check if all tasks are completed and update model selection availability
                    all_completed = all(st.session_state.task_completion[f"task{i}"]["correct"] for i in range(1, 6))
                    if all_completed:
                        st.session_state.can_select_model = True
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
            
            # Handle skip button action
            if can_skip and skip_button:
                # Mark task as skipped
                st.session_state.skipped_tasks.add(task_id)
                
                # Log the skip action
                log_user_answer(
                    st.session_state.username, 
                    task_id, 
                    "SKIPPED", 
                    False,
                    model_used=st.session_state.task_models[task_id],
                    query=f"Task skipped - {task_id}"  # Mark as a skipped task
                )
                
                # Show confirmation
                st.warning(f"Task {task_id[-1]} has been skipped. You can continue with other tasks.")
                
                # Check if all tasks are now completed or skipped
                if check_all_tasks_completed_or_skipped():
                    st.session_state.can_select_model = True
                    st.session_state.show_feedback_popup = True
                    st.success("All tasks complete! You can now choose which AI model to use.")
                    st.rerun()  # Rerun to show the popup
            
            # Display task status summary
            st.divider()
            st.subheader("Task Progress")
            
            for i in range(1, 6):
                task_key = f"task{i}"
                task_data = st.session_state.task_completion[task_key]
                
                if task_key in st.session_state.skipped_tasks:
                    status = "⏩ Skipped"
                    color = "orange"
                elif not task_data["completed"]:
                    status = "⚪ Not attempted"
                    color = "gray"
                elif task_data["correct"]:
                    status = "✅ Completed successfully"
                    color = "green"
                else:
                    status = f"❌ Attempted ({task_data['attempts']})"
                    color = "red"
                
                st.markdown(f"**Task {i}**: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)

            # Check if all tasks are completed successfully or skipped
            if check_all_tasks_completed_or_skipped() and not st.session_state.can_select_model:
                st.session_state.can_select_model = True
                st.session_state.show_feedback_popup = True
                # Show message about model selection being available
                st.success("🎉 All tasks completed! You can now choose which AI model to use.")

            # Show evaluation form if all tasks are completed successfully and evaluation not yet submitted
            if check_all_tasks_completed_or_skipped() and 'evaluation_submitted' not in st.session_state:
                st.session_state.show_evaluation_form = True

            # Display the evaluation form in a modal-like container
            if check_all_tasks_completed_or_skipped() and st.session_state.get('show_evaluation_form', False):
                st.markdown("### 🎉 Congratulations on completing all tasks!")
                
                with st.container():
                    st.markdown("""
                    ## Feedback Form
                    Please tell us what you think about the NASA Lessons Learned system.
                    Your feedback will help us improve it.
                    """)
                    
                    with st.form("evaluation_form"):
                        # Part 1: System Usability Scale (SUS) Questions
                        st.header("System Usability")
                        
                        # SUS Questions - Using 5-point Likert scale
                        sus_questions = [
                            "I think that I would like to use this system frequently.",
                            "I found the system unnecessarily complex.",
                            "I thought the system was easy to use.",
                            "I think that I would need the support of a technical person to be able to use this system.",
                            "I found the various functions in this system were well integrated.",
                            "I thought there was too much inconsistency in this system.",
                            "I would imagine that most people would learn to use this system very quickly.",
                            "I found the system very cumbersome to use.",
                            "I felt very confident using the system.",
                            "I needed to learn a lot of things before I could get going with this system."
                        ]
                        
                        sus_responses = {}
                        for i, question in enumerate(sus_questions, 1):
                            sus_responses[f"sus_q{i}"] = st.radio(
                                question,
                                options=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
                                horizontal=True,
                                key=f"sus_q{i}"
                            )
                        
                        # Task difficulty questions
                        st.header("Task Difficulty")
                        
                        # Create a more structured layout for task difficulty
                        task_difficulty = {}
                        for i in range(1, 6):
                            task_difficulty[f"task{i}"] = st.radio(
                                f"How difficult was Task {i}?",
                                options=["Too Easy", "Easy", "Just Right", "Challenging", "Too Difficult"],
                                horizontal=True,
                                key=f"difficulty_task{i}"
                            )
                        
                        # AI Assistant Performance
                        st.header("AI Assistant Performance")
                        
                        ai_helpfulness = st.radio(
                            "How helpful was the AI assistant?",
                            options=["Not helpful", "Slightly helpful", "Moderately helpful", "Very helpful", "Extremely helpful"],
                            horizontal=True
                        )
                        
                        ai_relevance = st.radio(
                            "How relevant were the AI's responses to your questions?",
                            options=["Not relevant", "Somewhat relevant", "Moderately relevant", "Very relevant", "Extremely relevant"],
                            horizontal=True
                        )
                        
                        # Research-Specific Questions
                        st.header("Research Features")
                        
                        retrieval_quality = st.radio(
                            "How would you rate the quality of information you found?",
                            options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
                            horizontal=True
                        )
                        
                        traditional_comparison = st.radio(
                            "Compared to regular search methods (Google, Bing, etc.), this system is:",
                            options=["Much worse", "Worse", "About the same", "Better", "Much better"],
                            horizontal=True
                        )
                        
                        # Open-ended feedback
                        st.header("Additional Feedback")
                        
                        improvement_suggestions = st.text_area(
                            "How can we improve this system?",
                            height=100
                        )
                        
                        favorite_feature = st.text_area(
                            "What was your favorite feature?",
                            height=100
                        )
                        
                        # Submit button
                        submitted = st.form_submit_button("Submit Feedback")
                        
                        if submitted:
                            # Collect all evaluation data
                            evaluation_data = {
                                # System Usability Scale responses
                                "sus_responses": sus_responses,
                                
                                # Task difficulty
                                "task_difficulty": task_difficulty,
                                
                                # AI assistant evaluation
                                "ai_helpfulness": ai_helpfulness,
                                "ai_relevance": ai_relevance,
                                
                                # Research-specific evaluation
                                "retrieval_quality": retrieval_quality,
                                "traditional_comparison": traditional_comparison,
                                
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

    with tab1:
        st.header("NASA Mission Knowledge Base")
        
        # Model selection with conditional enabling
        model_col1, model_col2 = st.columns(2)
        with model_col1:
            st.write("Select AI Model:")
        with model_col2:
            model_options = ["OpenAI", "Llama"]
            
            # Only allow model selection after all tasks are completed
            if st.session_state.can_select_model:
                selected_index = 0 if st.session_state.selected_model == "openai" else 1
                selected_model = st.selectbox(
                    "Model",
                    model_options,
                    index=selected_index,
                    label_visibility="collapsed"
                )
                st.session_state.selected_model = selected_model.lower()
            else:
                # Show disabled dropdown with info message
                st.selectbox(
                    "Model",
                    model_options,
                    disabled=True,
                    label_visibility="collapsed"
                )
                st.info("Complete all tasks to select a model. Currently using randomly assigned models per task.")
        
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
            
            # Use the current task ID from session state instead of trying to determine it here
            current_task_id = st.session_state.current_task_id
            
            # Generate model name for display based on whether model selection is enabled
            if st.session_state.can_select_model:
                display_model = st.session_state.selected_model.upper()
            else:
                # If a task is active, use the model assigned to that task
                display_model = (st.session_state.task_models[current_task_id].upper() 
                                 if current_task_id 
                                 else st.session_state.selected_model.upper())
            
            # Get response from RAG model
            with st.spinner(f"Thinking using {display_model}..."):
                response = process_rag_query(rag_query, current_task_id)
            
            # Add assistant response to chat history
            st.session_state.rag_messages.append({"role": "assistant", "content": response})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response)

# Display feedback form as a popup when all tasks are completed/skipped
if st.session_state.show_feedback_popup and 'evaluation_submitted' not in st.session_state:
    # Create a custom dialog-like interface instead of using st.dialog()
    feedback_container = st.container()
    
    with feedback_container:
        # Add a colored background container to make it stand out
        with st.container(border=True):
            st.markdown("## 🎉 Feedback Form - Please complete before continuing")
            st.markdown("### Thank you for completing the tasks!")
            
            with st.form("popup_evaluation_form"):
                # Part 1: System Usability Scale (SUS) Questions
                st.header("System Usability")
                
                # SUS Questions - Using 5-point Likert scale
                sus_questions = [
                    "I think that I would like to use this system frequently.",
                    "I found the system unnecessarily complex.",
                    "I thought the system was easy to use.",
                    "I think that I would need the support of a technical person to be able to use this system.",
                    "I found the various functions in this system were well integrated.",
                    "I thought there was too much inconsistency in this system.",
                    "I would imagine that most people would learn to use this system very quickly.",
                    "I found the system very cumbersome to use.",
                    "I felt very confident using the system.",
                    "I needed to learn a lot of things before I could get going with this system."
                ]
                
                sus_responses = {}
                for i, question in enumerate(sus_questions, 1):
                    sus_responses[f"sus_q{i}"] = st.radio(
                        question,
                        options=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
                        horizontal=True,
                        key=f"popup_sus_q{i}"
                    )
                
                # Task difficulty questions
                st.header("Task Difficulty")
                
                # Create a more structured layout for task difficulty
                task_difficulty = {}
                for i in range(1, 6):
                    task_key = f"task{i}"
                    if task_key in st.session_state.skipped_tasks:
                        # If task was skipped, mark as "Too Difficult" by default, but allow changing
                        difficulty_options = ["Too Easy", "Easy", "Just Right", "Challenging", "Too Difficult"]
                        default_idx = 4  # "Too Difficult"
                        task_difficulty[task_key] = st.radio(
                            f"How difficult was Task {i}? (Skipped)",
                            options=difficulty_options,
                            index=default_idx,
                            horizontal=True,
                            key=f"popup_difficulty_task{i}"
                        )
                    else:
                        task_difficulty[task_key] = st.radio(
                            f"How difficult was Task {i}?",
                            options=["Too Easy", "Easy", "Just Right", "Challenging", "Too Difficult"],
                            horizontal=True,
                            key=f"popup_difficulty_task{i}"
                        )
                
                # AI Assistant Performance
                st.header("AI Assistant Performance")
                
                ai_helpfulness = st.radio(
                    "How helpful was the AI assistant?",
                    options=["Not helpful", "Slightly helpful", "Moderately helpful", "Very helpful", "Extremely helpful"],
                    horizontal=True,
                    key="popup_ai_helpfulness"
                )
                
                ai_relevance = st.radio(
                    "How relevant were the AI's responses to your questions?",
                    options=["Not relevant", "Somewhat relevant", "Moderately relevant", "Very relevant", "Extremely relevant"],
                    horizontal=True,
                    key="popup_ai_relevance"
                )
                
                # Research-Specific Questions
                st.header("Research Features")
                
                retrieval_quality = st.radio(
                    "How would you rate the quality of information you found?",
                    options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
                    horizontal=True,
                    key="popup_retrieval_quality"
                )
                
                traditional_comparison = st.radio(
                    "Compared to regular search methods (Google, Bing, etc.), this system is:",
                    options=["Much worse", "Worse", "About the same", "Better", "Much better"],
                    horizontal=True,
                    key="popup_traditional_comparison"
                )
                
                # Open-ended feedback
                st.header("Additional Feedback")
                
                improvement_suggestions = st.text_area(
                    "How can we improve this system?",
                    height=100,
                    key="popup_improvement_suggestions"
                )
                
                favorite_feature = st.text_area(
                    "What was your favorite feature?",
                    height=100,
                    key="popup_favorite_feature"
                )
                
                # Add skipped tasks information
                skipped_tasks_list = list(st.session_state.skipped_tasks)
                
                # Submit button
                submitted = st.form_submit_button("Submit Feedback")
                
                if submitted:
                    # Collect all evaluation data
                    evaluation_data = {
                        # System Usability Scale responses
                        "sus_responses": sus_responses,
                        
                        # Task difficulty
                        "task_difficulty": task_difficulty,
                        
                        # AI assistant evaluation
                        "ai_helpfulness": ai_helpfulness,
                        "ai_relevance": ai_relevance,
                        
                        # Research-specific evaluation
                        "retrieval_quality": retrieval_quality,
                        "traditional_comparison": traditional_comparison,
                        
                        # Open-ended feedback
                        "improvement_suggestions": improvement_suggestions,
                        "favorite_feature": favorite_feature,
                        
                        # Add information about skipped tasks
                        "skipped_tasks": skipped_tasks_list
                    }
                    
                    # Log the evaluation
                    log_user_evaluation(st.session_state.username, evaluation_data)
                    
                    # Update session state
                    st.session_state.evaluation_submitted = True
                    st.session_state.show_feedback_popup = False
                    
                    # Show success message
                    st.success("Thank you for your feedback! Your evaluation has been submitted successfully.")
                    st.balloons()
                    
                    # Rerun to close the feedback form
                    st.rerun()
