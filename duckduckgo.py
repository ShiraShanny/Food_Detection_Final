from io import BytesIO
import re
import requests
from duckduckgo_search import DDGS  # DuckDuckGo search library
import streamlit as st
from PIL import Image

# Function to search for images using DuckDuckGo
def search_images(term, max_images=5):
    with DDGS() as ddgs:
        ddgs_images_gen = ddgs.images(keywords=term, max_results=max_images)
        return [result['image'] for result in ddgs_images_gen]

# Function to sanitize filenames (removes invalid characters)
def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)

# Function to download and show an image
def display_image_from_url(url, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # If the request is successful, display the image
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="Recipe Image", use_container_width=True)
                return  # Exit function after displaying image
        except Exception as e:
           print("problem download image")
