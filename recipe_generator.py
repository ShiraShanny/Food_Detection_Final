from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import requests
import re
from duckduckgo_search import DDGS
from pathlib import Path

# Set the API key and initialize the language model
GROQ_API_KEY = "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"
llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=GROQ_API_KEY)

# Define data models for the recipe
class Ingredient(BaseModel):
    name: str
    amount: str
    unit: str

class Recipe(BaseModel):
    title: str
    servings: int = Field(..., description="Number of servings")  # Now required
    prep_time: str = Field(..., description="Preparation time")  # Now required
    cook_time: str = Field(..., description="Cooking time")  # Now required
    total_time: str = Field(..., description="Total time")  # Now required
    difficulty: str = Field(..., description="Difficulty level")  # Now required
    cuisine: str = Field(..., description="Cuisine type")  # Now required
    category: str = Field(..., description="Category of the dish")  # Now required
    ingredients: List[Ingredient]
    equipment: List[str]
    instructions: List[str]
    tips: List[str]
    nutrition: Dict[str, Any] = Field(..., description="Nutritional information")  # Now required
    detected_products: List[str] = Field( description="List of detected products")
    removed_products: List[str] = Field(description="List of ingredients that were not included in the final recipe, indicating all ingredients present in the input list that were not retained.")
    added_products: List[str] = Field(description="List of ingredients that were added to the recipe, which were not part of the original input list but are necessary for the recipe.")


# Function to generate the recipe
def generate_recipe(ingredients_list: List[str]):
    ingredients_str = ", ".join(ingredients_list)

    # Create a prompt for the language model
    prompt = ChatPromptTemplate(
        [
            SystemMessagePromptTemplate.from_template(
                "The assistant is a chef and recipe expert. "
                "The task is to generate a recipe using the given ingredients. "
                "You must add any necessary ingredients, indicate which provided ingredients are not suitable, remove ingredients and added more 2 minumum must "
                "and suggest alternative ingredients if necessary. must adding or remove "
            ),
            HumanMessagePromptTemplate.from_template(f"""
            The ingredients provided are:

            {ingredients_list}

            Please generate a recipe in the specified format, ensuring to add and remove any missing ingredients required for the recipe.
            """)
        ]
    )
    recipe_chain = prompt | llm.with_structured_output(schema=Recipe)

    try:
        result = recipe_chain.invoke({
            "ingredients_list": ingredients_str
        })
        return result
    except Exception as e:
        return None

# Function to fetch an image for the recipe
def fetch_image_for_recipe(recipe_title: str):
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    with DDGS() as ddgs:
        ddgs_images_gen = ddgs.images(keywords=recipe_title, max_results=5)
        image_urls = [result['image'] for result in ddgs_images_gen]

    for image_url in image_urls:
        image_path = download_image(images_dir, image_url)
        if image_path:
            return image_path
    return None

def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def download_image(dest: Path, url: str):
    try:
        filename = sanitize_filename(url.split('/')[-1])
        response = requests.get(url)
        if response.status_code == 200:
            with open(dest / filename, 'wb') as f:
                f.write(response.content)
            return str(dest / filename)
        return None
    except Exception:
        return None
