import os
from typing import Dict, List, Any, TypedDict, Literal
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Import only the Tavily search tool
from utils import search_tavily, get_language_code

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "english")

# Define the state
class RecipeState(TypedDict):
    """Represents the state of the recipe generation process."""
    messages: List[Any]
    ingredients: str
    language: str
    search_results: str
    recipe: str
    next: Literal["search", "generate", "end"]

# System prompts
SEARCH_SYSTEM_PROMPT = """
You are a warm, friendly recipe search assistant named Rasoi Mitra (Kitchen Friend). Your personality is like a caring family member who loves to cook and share recipes. Your task is to find MANY diverse recipe ideas based on the ingredients the user has.

VERY IMPORTANT INSTRUCTIONS:
1. DO NOT include any thinking, reasoning, or explanations about how you found the recipes.
2. DO NOT use phrases like 'Let me think', 'I'll look for', 'First', 'Based on', etc.
3. Jump directly into your response without any preamble.
4. ALWAYS be enthusiastic and interactive throughout your response.
5. Make your response sound like you're excited to help them cook something delicious!

First, acknowledge the user's request in a friendly way, as if you're having a conversation with a family member. Then provide a concise summary of at least 5-7 different recipe ideas that can be made with the user's ingredients. Focus on variety and different cuisines.

If the user has asked any questions or made comments, respond to them in a conversational manner before providing recipe ideas.

If the language is set to Hindi, you should respond in Hinglish (a mix of Hindi and English) using both Hindi and Roman script as appropriate. This is more natural for many Indian users.
"""

GENERATOR_SYSTEM_PROMPT = """
You are a warm, friendly, and conversational recipe assistant named Rasoi Mitra (Kitchen Friend). Your personality is like a caring family member who loves to cook and share recipes. You are creating MULTIPLE detailed recipes (at least 5 different options) based on the ingredients and ideas provided, but you're also having a conversation with the user.

VERY IMPORTANT INSTRUCTIONS:
1. DO NOT include any thinking, reasoning, or explanations about how you created the recipes.
2. DO NOT use phrases like 'Let me think', 'I'll create', 'First', 'Based on', etc.
3. Jump directly into your response without any preamble.
4. ALWAYS end your response with an engaging follow-up question.
5. Be enthusiastic and interactive throughout your response.

First, respond to any questions or comments the user has made in a friendly, conversational way. Address them by name if they've shared it, and make them feel like they're talking to a real person who cares about their cooking journey.

Then, for EACH recipe, include:
1. A creative name for the dish
2. Complete list of ingredients with measurements
3. Step-by-step cooking instructions
4. Preparation and cooking time
5. Any tips or variations

Format each recipe with a clear heading and number (Recipe #1, Recipe #2, etc.) to make them easy to distinguish.

Be detailed, practical, and creative. Make sure the recipes are actually feasible with the ingredients provided. Try to provide variety in the types of dishes (e.g., breakfast, lunch, dinner, snacks, etc.).

At the end, ALWAYS ask an engaging follow-up question like "Which one would you like to try first?" or "Do you have any questions about these recipes?" to keep the conversation going. Make this follow-up question exciting and personalized.

If the language is set to Hindi, you should respond in Hinglish (a mix of Hindi and English) using both Hindi and Roman script as appropriate. This is more natural for many Indian users. For example, use terms like 'tadka', 'masala', etc. in Roman script along with Hindi script where appropriate.
"""

# Initialize the LLM
def get_llm(max_tokens=4000, temp=0.7):
    """Initialize and return the LLM with customizable parameters."""
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=temp,
        max_tokens=max_tokens,
    )

# Simplified search function that uses only Tavily
def search_for_recipes(ingredients: str, language: str, user_message: str = "") -> str:
    """Search for recipes using the ingredients and respond to user message."""
    try:
        llm = get_llm()
        
        print(f"Searching for recipes with ingredients: {ingredients} in {language}")
        
        # Create a search prompt that includes the user's message
        search_prompt = f"""
        User's message: {user_message}
        
        Find recipe ideas using these ingredients: {ingredients}.
        
        Language: {language}
        """
        
        # Use Tavily to search for recipes
        tavily_result = search_tavily(f"recipes with {ingredients}")
        print(f"Search completed. Result length: {len(tavily_result)}")
        
        # Combine the search results with the prompt
        search_prompt += f"""
        
        Here is some information from the web that might help:
        {tavily_result}
        
        Remember to respond to the user's message in a warm, friendly way before providing recipe ideas.
        """
        
        # Generate recipe ideas using the LLM
        messages = [
            SystemMessage(content=SEARCH_SYSTEM_PROMPT),
            HumanMessage(content=search_prompt)
        ]
        
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Error in search_for_recipes: {str(e)}")
        return f"I couldn't find specific recipes for {ingredients}, but I'll create a recipe for you anyway."

# Simplified recipe generation function
def generate_detailed_recipe(ingredients: str, language: str, search_results: str, user_message: str = "") -> str:
    """Generate multiple detailed recipes based on the ingredients and search results while responding to user's message."""
    try:
        # Use a higher max_tokens value to accommodate multiple recipes
        llm = get_llm(max_tokens=8000, temp=0.8)
        
        print(f"Generating multiple recipes for ingredients: {ingredients} in {language}")
        
        # Create a recipe generation prompt that includes the user's message
        recipe_prompt = f"""
        User's message: {user_message}
        
        Create 5 different detailed recipes using these ingredients: {ingredients}.
        
        Here are some recipe ideas to consider: {search_results}
        
        First, respond to the user's message in a warm, friendly, conversational way.
        Then, for EACH recipe, include:
        1. A creative name for the dish
        2. Complete list of ingredients with measurements
        3. Step-by-step cooking instructions
        4. Preparation and cooking time
        5. Any tips or variations
        
        Number each recipe clearly (Recipe #1, Recipe #2, etc.) and make them distinct from each other.
        Provide variety in cooking styles, cuisines, and difficulty levels.
        
        At the end, ask a follow-up question to keep the conversation going.
        
        Language: {language}
        """
        
        # Create the recipe prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=recipe_prompt)
        ])
        
        # Create the recipe generation chain
        recipe_chain = prompt | llm | StrOutputParser()
        
        # Generate the recipe
        recipe = recipe_chain.invoke({})
        
        # Ensure there's a follow-up question at the end
        if not any(question in recipe.lower() for question in ["?", "which one", "what do you think", "would you like", "do you have"]):
            # Add a follow-up question if none exists
            follow_up_questions = [
                "Which recipe sounds most delicious to you? 😋",
                "Would you like me to explain any of these recipes in more detail? 🍽️",
                "Which one would you like to try first? 👨‍🍳",
                "Do any of these recipes catch your eye? Let me know! ✨",
                "Are there any ingredients you'd like to substitute in these recipes? 🥕"
            ]
            import random
            recipe += f"\n\n{random.choice(follow_up_questions)}"
        
        print(f"Recipe generated successfully. Length: {len(recipe)}")
        
        return recipe
    except Exception as e:
        print(f"Error in generate_detailed_recipe: {str(e)}")
        return create_fallback_recipe(ingredients, language)

# Fallback recipe function
def create_fallback_recipe(ingredients: str, language: str) -> str:
    """Create a fallback recipe when the LLM fails."""
    # Extract main ingredients from the input
    common_ingredients = ["potato", "onion", "tomato", "chicken", "rice", "paneer", "dal", "egg"]
    found_ingredients = []
    
    for ingredient in common_ingredients:
        if ingredient in ingredients.lower():
            found_ingredients.append(ingredient)
    
    # If no ingredients found, use default ones
    if not found_ingredients:
        found_ingredients = ["potato", "onion", "tomato"]
    
    # Create a fallback recipe based on the found ingredients
    if language.lower() == "hindi":
        # Hinglish fallback recipe
        return f"""Recipe: {', '.join(found_ingredients).title()} ki Sabzi

Samagri (Ingredients):
- {', '.join(found_ingredients).title()}
- Haldi, Dhaniya powder, Garam masala
- Namak swadanusar
- Cooking oil

Banane ki Vidhi (Instructions):
1. Sabhi vegetables ko cut karein
2. Kadhai mein oil garam karein
3. Masale daale aur vegetables add karein
4. Dhak kar 15 minute tak pakayein
5. Garam garam serve karein

Taiyari ka samay: 10 minute
Pakane ka samay: 15 minute"""
    else:
        return f"""Recipe: Simple {', '.join(found_ingredients).title()} Stir-Fry

Ingredients:
- {', '.join(found_ingredients).title()}
- Salt and pepper to taste
- Mixed herbs and spices
- Cooking oil

Instructions:
1. Prepare all vegetables by washing and cutting them
2. Heat oil in a pan over medium heat
3. Add the ingredients and spices
4. Cook for about 15 minutes, stirring occasionally
5. Serve hot with your choice of bread or rice

Prep Time: 10 minutes
Cook Time: 15 minutes"""

# Node definitions for the graph
def search_node(state: RecipeState) -> Dict[str, Any]:
    """Search for recipes based on the ingredients."""
    try:
        # Extract ingredients and language from the state
        ingredients = state["ingredients"]
        language = state["language"]
        user_message = state.get("user_message", ingredients)  # Use ingredients as fallback
        
        print(f"Searching for recipes with ingredients: {ingredients} in {language}")
        
        # Search for recipes
        search_results = search_for_recipes(ingredients, language, user_message)
        
        # Update the state with the search results
        state["search_results"] = search_results
        state["next"] = "generate"
        
        return state
    except Exception as e:
        print(f"Error in search_node: {str(e)}")
        # If there's an error, move to recipe generation with empty search results
        state["search_results"] = ""
        state["next"] = "generate"
        return state

def generate_node(state: RecipeState) -> Dict[str, Any]:
    """Generate a recipe based on the search results."""
    ingredients = state.get("ingredients", "")
    language = state.get("language", "english")
    search_results = state.get("search_results", "")
    user_message = state.get("user_message", "")
    
    # Generate the recipe
    recipe = generate_detailed_recipe(ingredients, language, search_results, user_message)
    
    # Add the recipe to the messages
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=recipe))
    
    # Return the updated state
    return {"recipe": recipe, "messages": messages, "next": "end"}

# Create the graph
def create_recipe_graph():
    """Create and return the recipe generation graph."""
    # Define the graph
    workflow = StateGraph(RecipeState)
    
    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("generate", generate_node)
    
    # Add edges
    workflow.add_edge("search", "generate")
    workflow.add_edge("generate", END)
    
    # Set the entry point
    workflow.set_entry_point("search")
    
    # Compile the graph
    return workflow.compile()

# Initialize the graph
recipe_graph = create_recipe_graph()

# Function to process user input
def process_user_input(user_input: str, conversation_history: List[Any] = None) -> str:
    """Process user input and generate a recipe response."""
    try:
        if conversation_history is None:
            conversation_history = []
        
        print(f"Processing input: {user_input}")
        
        # Extract language preference from input
        language = DEFAULT_LANGUAGE
        if "hindi" in user_input.lower():
            language = "hindi"
        print(f"Language: {language}")
        
        # Create a system message
        system_message = SystemMessage(content="You are a helpful recipe assistant that helps users find recipes based on ingredients they have at home. Provide ONLY the recipe without any additional thinking or reasoning.")
        
        # Create a human message from the user input
        human_message = HumanMessage(content=user_input)
        
        # Combine messages
        messages = [system_message] + conversation_history + [human_message]
        print(f"Number of messages in history: {len(messages)}")
        
        # Create the initial state
        initial_state = {
            "messages": messages,
            "ingredients": user_input,  # For simplicity, we use the entire input as ingredients
            "language": language,
            "search_results": "",
            "recipe": "",
            "next": "search"
        }
        
        # Invoke the graph
        result = recipe_graph.invoke(initial_state)
        
        # Extract the response
        if result and "messages" in result and result["messages"]:
            # Get the last message as the response
            response = result["messages"][-1].content
            
            # No filtering needed here as we're using the process_user_input.py file now
            
            return response
        else:
            # Fallback response
            return create_fallback_recipe(user_input, language)
    except Exception as e:
        print(f"Error in process_user_input: {str(e)}")
        return f"I'm sorry, I encountered an error while processing your request. Please try again with different ingredients."
