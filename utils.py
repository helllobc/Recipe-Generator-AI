import os
import arxiv
import wikipedia
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
# Using standard wikipedia package instead of WikipediaAPIWrapper
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure API keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Language support
SUPPORTED_LANGUAGES = {
    "english": "en",
    "hindi": "hi"
}

DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "english")

# Tool schemas
class RecipeIngredientQuery(BaseModel):
    """Input for recipe ingredient search."""
    ingredients: str = Field(description="List of ingredients available to the user")
    language: str = Field(description="Language for the response (english or hindi)")

class ArxivSearchInput(BaseModel):
    """Input for ArXiv search."""
    query: str = Field(description="Search query for ArXiv")
    max_results: int = Field(description="Maximum number of results to return", default=5)

class WikipediaSearchInput(BaseModel):
    """Input for Wikipedia search."""
    query: str = Field(description="Search query for Wikipedia")
    language: str = Field(description="Language code (en, hi, etc.)", default="en")

class TavilySearchInput(BaseModel):
    """Input for Tavily search."""
    query: str = Field(description="Search query for Tavily")

# Tool implementations
@tool(args_schema=ArxivSearchInput)
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search ArXiv for academic papers on food and recipes."""
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = []
        for paper in search.results():
            results.append({
                "title": paper.title,
                "authors": ", ".join(author.name for author in paper.authors),
                "summary": paper.summary,
                "published": paper.published.strftime("%Y-%m-%d"),
                "url": paper.pdf_url
            })
        
        if not results:
            return "No relevant papers found on ArXiv."
        
        formatted_results = "\n\n".join(
            f"Title: {r['title']}\nAuthors: {r['authors']}\n"
            f"Published: {r['published']}\nSummary: {r['summary']}\n"
            f"URL: {r['url']}"
            for r in results
        )
        return formatted_results
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"

@tool(args_schema=WikipediaSearchInput)
def search_wikipedia(query: str, language: str = "en") -> str:
    """Search Wikipedia for information on food, ingredients, and recipes."""
    try:
        # Set Wikipedia language
        wikipedia.set_lang(language)
        
        # Search for the query
        search_results = wikipedia.search(query, results=3)
        
        if not search_results:
            return f"No Wikipedia results found for '{query}'."
        
        # Get the page for the first result
        try:
            page = wikipedia.page(search_results[0])
            return f"Title: {page.title}\n\nSummary: {page.summary}\n\nURL: {page.url}"
        except wikipedia.DisambiguationError as e:
            # If there's a disambiguation, get the first option
            try:
                page = wikipedia.page(e.options[0])
                return f"Title: {page.title}\n\nSummary: {page.summary}\n\nURL: {page.url}"
            except:
                return f"Multiple Wikipedia results found for '{query}', but couldn't retrieve details."
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

@tool(args_schema=TavilySearchInput)
def search_tavily(query: str) -> str:
    """Search the web for recipe information using Tavily."""
    try:
        if not TAVILY_API_KEY:
            return "Tavily API key not configured. Please set the TAVILY_API_KEY environment variable."
        
        client = TavilyClient(api_key=TAVILY_API_KEY)
        search_result = client.search(query=query, search_depth="advanced")
        
        if not search_result.get("results", []):
            return f"No web results found for '{query}'."
        
        formatted_results = "\n\n".join(
            f"Title: {result['title']}\n"
            f"Content: {result['content']}\n"
            f"URL: {result['url']}"
            for result in search_result.get("results", [])[:5]
        )
        
        return formatted_results
    except Exception as e:
        return f"Error searching Tavily: {str(e)}"

# Translation helpers
def get_language_code(language: str) -> str:
    """Convert language name to language code."""
    return SUPPORTED_LANGUAGES.get(language.lower(), SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE])

# Recipe-specific utilities
def format_recipe(recipe_data: Dict[str, Any], language: str = "english") -> str:
    """Format recipe data into a readable string."""
    if language.lower() == "hindi":
        return f"रेसिपी: {recipe_data['name']}\n\n" \
               f"सामग्री:\n{recipe_data['ingredients']}\n\n" \
               f"निर्देश:\n{recipe_data['instructions']}\n\n" \
               f"तैयारी का समय: {recipe_data['prep_time']}\n" \
               f"पकाने का समय: {recipe_data['cook_time']}"
    else:
        return f"Recipe: {recipe_data['name']}\n\n" \
               f"Ingredients:\n{recipe_data['ingredients']}\n\n" \
               f"Instructions:\n{recipe_data['instructions']}\n\n" \
               f"Prep Time: {recipe_data['prep_time']}\n" \
               f"Cook Time: {recipe_data['cook_time']}"
