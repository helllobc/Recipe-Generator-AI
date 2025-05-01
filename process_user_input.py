def process_user_input(user_input, conversation_history=None):
    """
    Process user input and generate a recipe response.
    Filters out thinking parts and only returns the actual recipe.
    Handles follow-up questions about specific recipes.
    """
    from recipe_graph import recipe_graph, create_fallback_recipe
    import re
    from langchain_core.messages import SystemMessage, HumanMessage
    import os
    from langchain_groq import ChatGroq
    
    # Get default language from environment or use English
    DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "english")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")
    
    try:
        if conversation_history is None:
            conversation_history = []
        
        print(f"Processing input: {user_input}")
        
        # Check if this is a follow-up question about a specific recipe
        is_follow_up = False
        previous_recipes = ""
        
        if user_input.startswith("FOLLOW_UP_QUESTION:"):
            is_follow_up = True
            # Extract the actual question and previous recipes
            parts = user_input.split("\n\nPREVIOUS_RECIPES: ", 1)
            if len(parts) == 2:
                user_input = parts[0].replace("FOLLOW_UP_QUESTION: ", "")
                previous_recipes = parts[1]
        
        # If it's a follow-up question, handle it differently
        if is_follow_up:
            # Initialize the LLM
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=MODEL_NAME,
                temperature=0.7,
                max_tokens=2000,
            )
            
            # Create a system message for answering follow-up questions
            system_content = """
            You are Rasoi Mitra (Kitchen Friend), a warm and helpful recipe assistant.
            The user is asking a follow-up question about one of the recipes you previously provided.
            Answer their question specifically about the recipe they're referring to.
            DO NOT generate new recipes unless explicitly asked to do so.
            Be conversational, friendly, and helpful.
            
            VERY IMPORTANT: DO NOT include any thinking, reasoning, or explanations about how you created the recipes. DO NOT use phrases like 'Let me think', 'I'll create', 'First', 'Based on', etc. Jump directly into your response without any preamble.
            
            Always end your response with a follow-up question to keep the conversation going.
            """
            
            # Create the prompt
            prompt = f"""
            The user previously received these recipes:
            
            {previous_recipes}
            
            Now they're asking: {user_input}
            
            Please answer their specific question about the recipe they're referring to.
            If they mention a specific recipe number (like 5th recipe), focus on that recipe.
            If they use Hindi/Hinglish words like 'bnani' (to make), provide detailed help for that specific recipe.
            Do not generate new recipes unless they explicitly ask for more recipes.
            Always end with a follow-up question to keep the conversation going.
            """
            
            # Generate the response
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            return response.content
        
        # For regular recipe requests (not follow-ups):
        # Extract ingredients from user input
        # Try to identify ingredients by looking for common food items
        # This is a simple approach - we'll extract the first part of the message
        # that seems to contain food items and use the rest as conversation
        
        # Common ingredient indicators
        ingredient_indicators = ["with", "using", "from", "make", "cook", "recipe", "dish", "food"]
        
        # Default to using the whole input as ingredients
        ingredients = user_input
        user_message = user_input
        
        # Extract language preference from input - support for Hinglish
        language = DEFAULT_LANGUAGE
        # Check for Hindi or Hinglish indicators
        hindi_indicators = ["hindi", "हिंदी", "hinglish", "हिंग्लिश"]
        if any(indicator in user_input.lower() for indicator in hindi_indicators):
            language = "hindi"
        # Also detect Hindi script characters
        if any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in user_input):
            language = "hindi"
        print(f"Language: {language}")
        
        # Create a system message with explicit instruction to be conversational
        system_message = SystemMessage(content="You are a warm, friendly recipe assistant named Rasoi Mitra (Kitchen Friend) that helps users find recipes based on ingredients they have at home. Respond to the user's questions and comments in a conversational way while providing detailed recipes.")
        
        # Create a human message from the user input
        human_message = HumanMessage(content=user_input)
        
        # Combine messages
        messages = [system_message] + conversation_history + [human_message]
        print(f"Number of messages in history: {len(messages)}")
        
        # Create the initial state
        initial_state = {
            "messages": messages,
            "ingredients": ingredients,  # The extracted ingredients
            "user_message": user_message,  # The full user message for conversation
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
            full_response = result["messages"][-1].content
            
            # Enhanced filter to remove thinking parts
            # Method 1: Check for explicit think/thinking markers
            if "</think>" in full_response:
                clean_response = full_response.split("</think>", 1)[1].strip()
            elif "think>" in full_response:
                clean_response = full_response.split("think>", 1)[1].strip()
            else:
                # Method 2: Use regex to find and remove thinking paragraphs
                thinking_phrases = [
                    "I'll", "Let me", "I think", "First", "Based on", "Here's", "I'd", "I would", 
                    "Wait", "Hmm", "Looking at", "Alternatively", "Maybe", "Let's", "I can", 
                    "I need to", "I should", "I could", "I'm going to", "I am going to"
                ]
                
                # First attempt: Look for recipe markers
                recipe_patterns = [
                    r"Recipe #1", r"Recipe #2", r"Recipe #3", r"Recipe #4", r"Recipe #5",
                    r"Recipe 1:", r"Recipe 2:", r"Recipe 3:", r"Recipe 4:", r"Recipe 5:",
                    r"\*\*Recipe 1\*\*", r"\*\*Recipe 2\*\*", r"\*\*Recipe 3\*\*", 
                    r"\*\*Recipe #1\*\*", r"\*\*Recipe #2\*\*", r"\*\*Recipe #3\*\*"
                ]
                
                for pattern in recipe_patterns:
                    match = re.search(pattern, full_response)
                    if match:
                        # Found a recipe marker, extract from there
                        clean_response = full_response[match.start():]
                        break
                else:
                    # Second attempt: Split by paragraphs and check for thinking phrases
                    if "\n\n" in full_response:
                        paragraphs = full_response.split("\n\n")
                        # Check each paragraph for thinking phrases
                        filtered_paragraphs = []
                        skip_paragraph = False
                        
                        for i, para in enumerate(paragraphs):
                            # Skip paragraphs that contain thinking phrases
                            if any(phrase in para for phrase in thinking_phrases):
                                skip_paragraph = True
                                continue
                            # If we've skipped a paragraph, check if this one starts a recipe
                            if skip_paragraph and any(pattern in para for pattern in ["Recipe", "Ingredients", "Instructions"]):
                                skip_paragraph = False
                            # Add non-thinking paragraphs
                            if not skip_paragraph:
                                filtered_paragraphs.append(para)
                        
                        if filtered_paragraphs:
                            clean_response = "\n\n".join(filtered_paragraphs)
                        else:
                            clean_response = full_response
                    else:
                        clean_response = full_response
                        
            # Final cleanup: Remove any remaining thinking phrases at the beginning of lines
            lines = clean_response.split("\n")
            filtered_lines = []
            
            # Define thinking phrases for final cleanup
            final_cleanup_phrases = [
                "I'll", "Let me", "I think", "First", "Based on", "Here's", "I'd", "I would", 
                "Wait", "Hmm", "Looking at", "Alternatively", "Maybe", "Let's", "I can", 
                "I need to", "I should", "I could", "I'm going to", "I am going to", "Now", "So", 
                "To create", "To make", "To prepare", "Let us", "We can", "We need to", "We should"
            ]
            
            for line in lines:
                if any(line.strip().startswith(phrase) for phrase in final_cleanup_phrases):
                    continue
                filtered_lines.append(line)
            
            clean_response = "\n".join(filtered_lines)
            
            # Ensure there's a follow-up question at the end
            if not any(question in clean_response[-100:].lower() for question in ["?", "which one", "what do you think", "would you like", "do you have"]):
                # Add a follow-up question if none exists
                follow_up_questions = [
                    "Which recipe sounds most delicious to you? 😋",
                    "Would you like me to explain any of these recipes in more detail? 🍽️",
                    "Which one would you like to try first? 👨‍🍳",
                    "Do any of these recipes catch your eye? Let me know! ✨",
                    "Are there any ingredients you'd like to substitute in these recipes? 🥕"
                ]
                import random
                clean_response += f"\n\n{random.choice(follow_up_questions)}"
            
            return clean_response
        else:
            # Fallback response
            return create_fallback_recipe(user_input, language)
    except Exception as e:
        print(f"Error in process_user_input: {str(e)}")
        return f"I'm sorry, I encountered an error while processing your request. Please try again with different ingredients."
