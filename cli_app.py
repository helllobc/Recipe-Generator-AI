import os
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from recipe_graph import process_user_input

# Load environment variables
load_dotenv()

# Check for API keys
if not os.getenv("GROQ_API_KEY"):
    print("Error: GROQ_API_KEY not found in environment variables.")
    print("Please create a .env file with your GROQ_API_KEY.")
    sys.exit(1)

def main():
    """Main CLI application for the recipe generator."""
    print("\n===== AI Recipe Generator =====\n")
    print("Welcome to the AI Recipe Generator!")
    print("Tell me what ingredients you have, and I'll suggest recipes you can make.")
    print("You can specify a language preference (English or Hindi).")
    print("Type 'exit' to quit.\n")
    
    conversation_history: List[Dict[str, Any]] = []
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nThank you for using the AI Recipe Generator. Goodbye!")
            break
        
        try:
            # Process the user input
            result = process_user_input(user_input, conversation_history)
            
            # Extract the latest AI message
            latest_message = next((msg for msg in reversed(result["messages"]) 
                                if isinstance(msg, AIMessage)), None)
            
            if latest_message:
                print(f"\nRecipe Assistant: {latest_message.content}")
            else:
                print("\nRecipe Assistant: I couldn't generate a recipe. Please try again.")
            
            # Update conversation history
            conversation_history = result["messages"]
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or check your API keys.")

if __name__ == "__main__":
    main()
