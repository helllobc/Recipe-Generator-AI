# AI Recipe Generator

## Overview

This AI Recipe Generator is a multilingual application that helps users discover recipes based on ingredients they have available at home. Using advanced AI powered by LangGraph and LangChain, the application can understand your available ingredients and generate detailed recipes in both English and Hindi.

## Features

- **Ingredient-Based Recipe Generation**: Simply tell the AI what ingredients you have, and it will suggest recipes you can make
- **Multilingual Support**: Get recipes in English or Hindi
- **Detailed Recipes**: Complete with ingredients, measurements, step-by-step instructions, and cooking times
- **Multiple Knowledge Sources**: Leverages ArXiv for academic food knowledge, Wikipedia for general culinary information, and Tavily for web search
- **Multiple Interfaces**: Use either the command-line interface or the user-friendly Streamlit web app

## Prerequisites

Before running the application, you'll need:

1. Python 3.9 or higher
2. A Groq API key (sign up at [groq.com](https://console.groq.com))
3. A Tavily API key (optional, for web search functionality)

## Installation

1. Clone this repository or download the files

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root directory with your API keys:

```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
MODEL_NAME=llama3-70b-8192
DEFAULT_LANGUAGE=english
```

## Usage

### Command Line Interface

To use the command-line interface:

```bash
python cli_app.py
```

Follow the prompts to enter your ingredients and get recipe recommendations.

### Streamlit Web Interface

For a more user-friendly experience, use the Streamlit web interface:

```bash
streamlit run streamlit_app.py
```

This will open a web browser with the application interface.

## Examples

### English Example

Input:
```
I have chicken, potatoes, onions, and tomatoes
```

The AI will generate a detailed recipe using these ingredients.

### Hindi Example

Input:
```
मेरे पास आलू, प्याज, टमाटर और मटर हैं (in hindi)
```

Or:
```
I have potatoes, onions, tomatoes, and peas (in hindi)
```

The AI will generate a recipe in Hindi using these ingredients.

## Project Structure

- `cli_app.py`: Command-line interface for the recipe generator
- `streamlit_app.py`: Streamlit web interface
- `recipe_graph.py`: Core LangGraph implementation for recipe generation
- `utils.py`: Utility functions and tools for knowledge retrieval
- `requirements.txt`: Dependencies for the project
- `.env.example`: Example environment variables configuration

## How It Works

The application uses a LangGraph workflow with the following steps:

1. **Routing**: Determines the next action based on user input
2. **Search**: Queries multiple knowledge sources (ArXiv, Wikipedia, Tavily) for recipe information
3. **Generation**: Creates detailed, personalized recipes based on the search results and user's ingredients

The system is powered by Groq's large language models, providing fast and accurate recipe generation.

## Troubleshooting

- **API Key Issues**: Ensure your Groq API key is correctly set in the `.env` file
- **Dependency Errors**: Make sure all required packages are installed via `pip install -r requirements.txt`
- **Language Support**: Currently only English and Hindi are supported

## License

This project is open source and available for personal and educational use.
