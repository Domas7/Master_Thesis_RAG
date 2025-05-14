import streamlit as st
import pandas as pd
from rag_models import get_rag_model
import datetime
import os
import json
import time
import random

# Initialize session state variables if they don't exist
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
if "last_used_model" not in st.session_state:
    st.session_state.last_used_model = "openai"  # Track which model was last used
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'completed_tasks' not in st.session_state:
    st.session_state.completed_tasks = set()
if 'all_tasks_completed' not in st.session_state:
    st.session_state.all_tasks_completed = False

# Initialize the RAG model
rag_model = get_rag_model()

# Add a function to print task model mappings for debugging
def print_task_model_mappings():
    """Print all task model mappings for debugging"""
    print("\n--- TASK MODEL MAPPINGS ---")
    if 'task_model_mapping' in st.session_state:
        for task_id, model in st.session_state.task_model_mapping.items():
            print(f"{task_id}: {model}")
    else:
        print("No task model mappings exist yet.")
    print("---------------------------\n")

# UI for the RAG tab
def render_rag_tab():
    st.header("NASA Mission Knowledge Base")
    
    # Model selection
    model_col1, model_col2 = st.columns(2)
    with model_col1:
        st.write("Select AI Model:")
    with model_col2:
        model_options = ["OpenAI", "Llama"]
        
        # Only allow model selection if all tasks are completed
        if st.session_state.all_tasks_completed:
            selected_index = 0 if st.session_state.selected_model == "openai" else 1
            selected_model = st.selectbox(
                "Model",
                model_options,
                index=selected_index,
                label_visibility="collapsed"
            )
            st.session_state.selected_model = selected_model.lower()
        else:
            # Disabled dropdown with explanation
            st.selectbox(
                "Model (Randomized)",
                model_options,
                disabled=True,
                label_visibility="collapsed"
            )
            st.info("📝 Models are being randomized by task until all evaluation tasks are completed or skipped.")
    
    # Chat interface for RAG
    if 'rag_messages' not in st.session_state:
        st.session_state.rag_messages = [
            {"role": "assistant", "content": "I can answer questions about NASA missions and documents. What would you like to know?"}
        ]
    
    # Display chat messages
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Display model info for assistant messages (except the first welcome message)
            if message["role"] == "assistant" and len(st.session_state.rag_messages) > 1 and message != st.session_state.rag_messages[0] and "model_used" in message:
                st.caption(f"Answered using: {message['model_used'].capitalize()}")
    
    # User input
    if rag_query := st.chat_input("Ask about NASA missions...", key="rag_input"):
        # Add user message to chat history
        st.session_state.rag_messages.append({"role": "user", "content": rag_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(rag_query)
        
        # Get response from RAG model
        with st.spinner(f"Thinking..."):
            response, model_used = process_rag_query(rag_query)
        
        # Add assistant response to chat history with model info
        st.session_state.rag_messages.append({
            "role": "assistant", 
            "content": response,
            "model_used": model_used
        })
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)

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

# Function to log user answers with additional information about model used and query
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
        "timestamp": timestamp
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

# Randomize model selection by task instead of by query
def get_model_for_task(task_id):
    """
    Get the model to use for a specific task.
    Each task always uses the same model, but the model is randomly assigned.
    """
    # Create a mapping of task to model if it doesn't exist
    if 'task_model_mapping' not in st.session_state:
        # Randomly assign models to tasks
        models = ["openai", "llama"]
        st.session_state.task_model_mapping = {
            f"task{i}": random.choice(models) for i in range(1, 6)
        }
        # Add a default for regular queries
        st.session_state.task_model_mapping["query"] = "openai"
        print("Initialized new task model mapping:", st.session_state.task_model_mapping)
    
    # If this task doesn't have a model assigned yet, assign one
    if task_id not in st.session_state.task_model_mapping:
        model = random.choice(["openai", "llama"])
        st.session_state.task_model_mapping[task_id] = model
        print(f"Assigned new model {model} for task {task_id}")
    
    # Return the model for this task
    return st.session_state.task_model_mapping.get(task_id, "openai")

# Process a query using the RAG model with model selected based on task
def process_rag_query(query, task_id="query"):
    """Process a query using the RAG model with model selected based on task"""
    # If all tasks are complete, use the selected model
    if st.session_state.all_tasks_completed:
        model = st.session_state.selected_model
        print(f"All tasks completed: Using user-selected model: {model}")
    else:
        # Use task-based model selection
        model = get_model_for_task(task_id)
        print(f"Task-based model selection: Using {model} for task {task_id}")
        st.session_state.last_used_model = model
    
    # Process the query
    print(f"Sending query to rag_model with model_name={model}")
    result = rag_model.query(query, model)
    
    # Log the query and model used
    if st.session_state.logged_in:
        log_user_answer(
            st.session_state.username, 
            "query", 
            result, 
            True,  # Not applicable for queries
            model_used=model,
            query=query
        )
    
    return result, model

# Check if all tasks are completed (either correct or skipped)
def check_all_tasks_completed():
    all_completed = True
    for i in range(1, 6):
        task_key = f"task{i}"
        if task_key not in st.session_state.task_completion:
            all_completed = False
            break
        task_data = st.session_state.task_completion[task_key]
        if not task_data.get("completed", False):
            all_completed = False
            break
    
    st.session_state.all_tasks_completed = all_completed
    return all_completed

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
        st.session_state.all_tasks_completed = False  # Reset task completion state
        st.session_state.task_completion = {
            "task1": {"completed": False, "correct": False, "attempts": 0},
            "task2": {"completed": False, "correct": False, "attempts": 0},
            "task3": {"completed": False, "correct": False, "attempts": 0},
            "task4": {"completed": False, "correct": False, "attempts": 0},
            "task5": {"completed": False, "correct": False, "attempts": 0}
        }
        st.session_state.completed_tasks = set()
        
        # Clear the task model mapping for a fresh start on next login
        if 'task_model_mapping' in st.session_state:
            del st.session_state.task_model_mapping
        
        st.rerun()
    
    # Display current user
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")

    # For admin only, show task model debugging
    if st.session_state.username == "admin":
        st.sidebar.write("---")
        st.sidebar.subheader("Task Model Assignments (Admin View)")
        # Display current task-model mappings
        for i in range(1, 6):
            task_key = f"task{i}"
            if task_key in st.session_state.task_model_mapping:
                model = st.session_state.task_model_mapping[task_key]
                st.sidebar.write(f"Task {i}: {model.capitalize()}")
            else:
                st.sidebar.write(f"Task {i}: Not yet assigned")
        
        # Add button to regenerate task models
        if st.sidebar.button("Reassign Task Models"):
            for i in range(1, 6):
                task_key = f"task{i}"
                st.session_state.task_model_mapping[task_key] = random.choice(["openai", "llama"])
            st.sidebar.success("Task models have been randomized!")
            st.rerun()
    
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
            
            # Show which model is being used for this task (admin only)
            if st.session_state.username == "admin":
                model_used = get_model_for_task(task_id)
                st.write(f"**Task Model:** {model_used.capitalize()}")

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
            
            # Create columns for submit and skip buttons
            col1, col2 = st.columns(2)
            
            with col1:
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
                    
                    # Get the model used for this task
                    model_used = get_model_for_task(task_id)
                    
                    # Log the user's answer with model information
                    log_user_answer(
                        st.session_state.username, 
                        task_id, 
                        user_answer, 
                        is_correct,
                        model_used=model_used, 
                        query=None  # No specific query for task submissions
                    )
                    
                    # Show feedback
                    if is_correct:
                        st.success(f"✅ Correct! {feedback_message}")
                        st.session_state.completed_tasks.add(task_id)
                        
                        # Check if all tasks are now completed
                        check_all_tasks_completed()
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
            
            with col2:
                # Initialize skip countdown state if not exists
                if f'skip_countdown_{task_id}' not in st.session_state:
                    st.session_state[f'skip_countdown_{task_id}'] = 3
                
                if f'skipping_{task_id}' not in st.session_state:
                    st.session_state[f'skipping_{task_id}'] = False
                
                if f'skip_confirmed_{task_id}' not in st.session_state:
                    st.session_state[f'skip_confirmed_{task_id}'] = False
                
                if f'countdown_complete_{task_id}' not in st.session_state:
                    st.session_state[f'countdown_complete_{task_id}'] = False
                
                # Skip button
                if not st.session_state[f'skipping_{task_id}'] and not st.session_state[f'countdown_complete_{task_id}']:
                    if st.button("Skip Task", key=f"skip_{task_id}"):
                        st.session_state[f'skipping_{task_id}'] = True
                
                # If skip button was pressed, show countdown
                if st.session_state[f'skipping_{task_id}'] and not st.session_state[f'countdown_complete_{task_id}']:
                    # Create a nicer popup-like container for countdown without using nested columns
                    st.markdown(f"""
                    <div style="padding: 15px; background-color: #f0f2f6; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; margin-bottom: 15px;">
                        <h3 style="color: #ff9800; margin-bottom: 10px;">⚠️ Skipping Task</h3>
                        <p style="margin-bottom: 15px;">Are you sure you want to skip this task? It's better to try first.</p>
                        <div style="font-size: 2.5rem; font-weight: bold; color: #ff5252; background-color: #ffebee; border-radius: 50%; width: 60px; height: 60px; line-height: 60px; margin: 0 auto 15px auto;">
                            {st.session_state[f'skip_countdown_{task_id}']}
                        </div>
                        <p style="font-size: 0.9rem; color: #757575;">Wait for countdown to complete...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Allow cancelling the skip
                    if st.button("Cancel Skip", key=f"cancel_skip_{task_id}"):
                        st.session_state[f'skipping_{task_id}'] = False
                        st.session_state[f'skip_countdown_{task_id}'] = 3
                        st.rerun()
                    
                    # Auto-decrease countdown and rerun
                    if st.session_state[f'skip_countdown_{task_id}'] > 0:
                        st.session_state[f'skip_countdown_{task_id}'] -= 1
                        # Wait for 1 second
                        time.sleep(1)
                        st.rerun()  # Rerun to update countdown
                    else:
                        # Countdown reached 0, show final confirmation
                        st.session_state[f'countdown_complete_{task_id}'] = True
                        st.session_state[f'skipping_{task_id}'] = False
                        st.rerun()
                
                # Final confirmation after countdown completes
                if st.session_state[f'countdown_complete_{task_id}'] and not st.session_state[f'skip_confirmed_{task_id}']:
                    st.markdown(f"""
                    <div style="padding: 15px; background-color: #fff3e0; border-radius: 10px; border: 1px solid #ffe0b2; text-align: center; margin-bottom: 15px;">
                        <h3 style="color: #e65100; margin-bottom: 10px;">Confirm Skip</h3>
                        <p>Click to confirm you want to skip this task.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Removed nested columns and used stacked buttons
                    confirm_skip = st.button("Yes, Skip Task", key=f"confirm_skip_{task_id}")
                    cancel_skip = st.button("No, I'll Try", key=f"cancel_confirm_skip_{task_id}")
                    
                    if confirm_skip:
                        st.session_state[f'skip_confirmed_{task_id}'] = True
                        
                        # Mark task as skipped (completed but with skipped status)
                        if task_id not in st.session_state.task_completion:
                            st.session_state.task_completion[task_id] = {"completed": False, "correct": False, "attempts": 0}
                        
                        st.session_state.task_completion[task_id]["completed"] = True
                        st.session_state.task_completion[task_id]["correct"] = True  # Mark as correct to allow progress
                        st.session_state.task_completion[task_id]["skipped"] = True  # Add skipped status
                        st.session_state.completed_tasks.add(task_id)
                        
                        # Get the model that would have been used for this task
                        model_used = get_model_for_task(task_id)
                        
                        # Log the skipped task with model information
                        log_user_answer(
                            st.session_state.username, 
                            task_id, 
                            "TASK SKIPPED", 
                            False,
                            model_used=model_used,
                            query=None
                        )
                        
                        # Check if all tasks are now completed after skipping
                        check_all_tasks_completed()
                        
                        # Show feedback
                        st.warning(f"Task {task_id[-1]} has been skipped. You can still try other tasks.")
                        st.session_state[f'countdown_complete_{task_id}'] = False
                        st.rerun()
                    
                    if cancel_skip:
                        st.session_state[f'countdown_complete_{task_id}'] = False
                        st.session_state[f'skip_countdown_{task_id}'] = 3
                        st.rerun()
            
            # Display task status summary
            st.divider()
            st.subheader("Task Progress")
            
            for i in range(1, 6):
                task_key = f"task{i}"
                task_data = st.session_state.task_completion[task_key]
                
                if not task_data["completed"]:
                    status = "⚪ Not attempted"
                    color = "gray"
                elif task_data.get("skipped", False):
                    status = "⏩ Skipped"
                    color = "orange"
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
        render_rag_tab()

# At the top of the app, print the current mappings
if st.session_state.logged_in:
    print_task_model_mappings()
