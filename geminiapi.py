import google.generativeai as genai
import streamlit as st
import ast

# gemmini

def detect_product_names(image_path):
    # Replace with your Google Cloud API key
    genai.configure(api_key="AIzaSyDJxR7uFT49FxrueweTPJO85PieypLdf0s")

    """Detects product names in an image using the Gemini API and returns them as a list.
       Returns:
        A list of detected product names or an empty list if none are found.
    """
    try:
        # Read the image
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        mime_type = f"image/{image_path.split('.')[-1]}"
        image_blob = {
            "mime_type": mime_type,
            "data": image_bytes
        }

        # Initialize the model
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Define the request
        request = {
            "parts": [
                {
                    "text": (
                        "Identify the individual products visible in this image. "
                        "Return the result as a Python list of unique product names. "
                        "Do not return the name of the list, only the list itself with the next format: ['item1','item2',....'item n']."
                    )
                },
                {"inline_data": image_blob},
            ]
        }

        # Generate content
        response = model.generate_content(request)

        # Debug: Write the full response to Streamlit

        # Parse the response
        if response.candidates:
            text_part = response.candidates[0].content.parts[0].text.strip()  # Strip whitespace/newlines

            # Safely parse the list
            try:
                product_names = ast.literal_eval(text_part)  # Use literal_eval for safety
                if isinstance(product_names, list):
                    return list(set(product_names))  # Deduplicate
            except (SyntaxError, ValueError, TypeError) as e:
                st.error(f"Failed to parse response: {e}")

        st.warning("No valid product list detected in the response.")
    except Exception as e:
        st.error(f"An error occurred during detection: {e}")
    return []
