import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from process_user_input import process_user_input
import re

# Load environment variables
load_dotenv()

# Function to detect if the user is asking about a specific recipe
def is_asking_about_recipe(user_input, previous_response):
    # If there's no previous response, return False
    if not previous_response:
        return False
        
    # Check if the previous response contains recipe headings
    has_recipes = 'Recipe #' in previous_response
    if not has_recipes:
        return False
    
    # Check for recipe references in various formats
    recipe_indicators = [
        # English patterns
        r'recipe\s*#?\s*[1-5]',  # Recipe #1, Recipe 1
        r'[1-5](?:st|nd|rd|th)\s+recipe',  # 1st recipe, 5th recipe
        r'#\s*[1-5]',  # #1, #5
        
        # Hindi/Hinglish patterns
        r'[1-5]\s*(?:th|st|nd|rd)?\s*recipe',  # 5th recipe, 5 recipe
        r'recipe\s*[1-5]',  # recipe 5
        
        # General reference patterns
        r'first recipe', r'second recipe', r'third recipe', r'fourth recipe', r'fifth recipe',
        r'last recipe', r'final recipe',
        
        # Direct number references when recipes are numbered
        r'\b[1-5]\b'  # Just the number 1-5 if recipes are numbered
    ]
    
    # Check if any of the patterns match
    for pattern in recipe_indicators:
        if re.search(pattern, user_input.lower()):
            return True
    
    # Additional checks for more general recipe references
    general_recipe_words = ['recipe', 'dish', 'meal', 'food', 'bnani', 'banane', 'khana', 'khaana', 'bana']
    if any(word in user_input.lower() for word in general_recipe_words):
        return True
        
    return False

# Check for API keys
if not os.getenv("GROQ_API_KEY"):
    st.error("Error: GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

# App title and description
st.set_page_config(page_title="AI Recipe Generator", page_icon="🍳", layout="wide")
st.title("🍳 AI Recipe Generator")

# Sidebar with app information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app helps you discover recipes based on ingredients you have at home.
    
    **How to use:**
    1. Enter the ingredients you have available
    2. Choose your preferred language
    3. Get personalized recipe recommendations
    
    **Features:**
    - Multilingual support (English & Hindi)
    - Detailed recipes with instructions
    - Powered by AI using LangGraph and LangChain
    """)
    
    st.header("Language")
    language = st.radio("Select your preferred language:", ["English", "Hindi"], index=0)
    
    st.header("Powered By")
    st.markdown("""
    - LangGraph & LangChain
    - Groq AI
    - ArXiv, Wikipedia & Tavily
    """)

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []
    system_message = SystemMessage(content="I am an AI recipe assistant that can help you find recipes based on ingredients you have at home.")
    st.session_state.conversation.append(system_message)

# Display conversation history
for message in st.session_state.conversation:
    if isinstance(message, SystemMessage):
        continue
    elif isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").write(message.content)

# User input
with st.container():
    # Input for ingredients
    user_input = st.chat_input("Enter the ingredients you have available...")
    
    if user_input:
        # Add language preference if selected
        if language.lower() == "hindi":
            full_input = f"{user_input} (in hindi)"
        else:
            full_input = user_input
        
        # Add user message to conversation
        user_message = HumanMessage(content=full_input)
        st.session_state.conversation.append(user_message)
        st.chat_message("user").write(full_input)
        
        # Show a spinner while processing
        with st.spinner("Generating recipe recommendations..."):
            try:
                # Check if the user is asking about a specific recipe from previous responses
                if len(st.session_state.conversation) >= 2 and is_asking_about_recipe(full_input, st.session_state.conversation[-2].content if isinstance(st.session_state.conversation[-2], AIMessage) else ""):
                    st.info("Answering your question about the recipe...")
                    
                    # Get the previous AI response that contains the recipes
                    previous_response = st.session_state.conversation[-2].content if isinstance(st.session_state.conversation[-2], AIMessage) else ""
                    
                    # Add a special flag to indicate this is a follow-up question about a recipe
                    full_input_with_context = f"FOLLOW_UP_QUESTION: {full_input}\n\nPREVIOUS_RECIPES: {previous_response}"
                    
                    # Process the follow-up question
                    result = process_user_input(full_input_with_context, st.session_state.conversation)
                else:
                    # Process the user input as a new recipe request
                    st.info("Processing your ingredients...")
                    result = process_user_input(full_input, st.session_state.conversation)
                
                # In the new version, result is a string containing the recipe
                if result:
                    # Create an AI message from the result
                    ai_message = AIMessage(content=result)
                    
                    # Add AI message to conversation
                    st.session_state.conversation.append(ai_message)
                    st.chat_message("assistant").write(result)
                else:
                    st.error("I couldn't generate a recipe. Please try again with more specific ingredients.")
                    st.info("Please try entering your ingredients again, for example: 'I have chicken, rice, and vegetables'.")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("Please try again or check your API keys.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using LangGraph and LangChain")
