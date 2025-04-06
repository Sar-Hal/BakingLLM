# Install required packages
#!pip install fastapi uvicorn python-multipart transformers google-generativeai pyngrok nest-asyncio
#!ngrok authtoken <>  # Replace with your actual token from ngrok.com

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import google.generativeai as genai
import os
import re
from fractions import Fraction
import json
from pyngrok import ngrok
import uvicorn
import nest_asyncio

nest_asyncio.apply()

# Load environment variables (for Kaggle, use os.environ directly)
GEMINI_API_KEY=""  # Replace with your actual key

# Initialize models
food_ner = pipeline(
    "token-classification",
    model="Dizex/InstaFoodRoBERTa-NER",
    aggregation_strategy="simple",
    device=-1
)
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI()

class RecipeRequest(BaseModel):
    text: str

def parse_quantity(qty_str):
    try:
        if ' ' in qty_str and '/' in qty_str:
            whole, fraction = qty_str.split()
            return float(whole) + float(Fraction(fraction))
        return float(Fraction(qty_str))
    except:
        return None

def extract_ingredients(text):
    unit_pattern = r'\b(cups?|tbsps?|tsps?|oz|lbs?|teaspoons?|tablespoons?|g|kg|ml)\b'
    ingredients = []

    for match in re.finditer(
        r'-\s*((\d+/\d+|\d+\.\d+|\d+\s\d+/\d+|\d+)\s*([a-zA-Z]+)\s+.*?)(?=,|\n|$)',
        text,
        re.IGNORECASE
    ):
        full_match = match.group(1)
        parts = [p.strip() for p in re.split(r',\s*(?=\d)', full_match)]

        for part in parts:
            match = re.search(
                r'^(\d+/\d+|\d+\.\d+|\d+\s\d+/\d+|\d+)\s*([a-zA-Z]+)\s*(.*)',
                part,
                re.IGNORECASE
            )
            if not match:
                continue

            qty, unit, ingredient_part = match.groups()
            unit_match = re.search(unit_pattern, unit.rstrip('s'), re.IGNORECASE)
            if not unit_match:
                continue
                
            unit = unit_match.group().lower()
            quantity = parse_quantity(qty)

            if quantity and unit and ingredient_part:
                ingredients.append({
                    "ingredient": ingredient_part.lower().replace('-', ' ').strip(),
                    "quantity": quantity,
                    "unit": unit
                })
    return ingredients

def convert_with_gemini(ingredients):
    prompt = f"""
    Convert these baking ingredients to PRECISE grams or milliliters. 
    Return ONLY this JSON format (no other text/comments):
    {{
      "results": [
        {{
          "ingredient": string,
          "grams": number (for dry ingredients),
          "ml": number (for liquids)
        }}
      ]
    }}

    RULES:
    1. Use EXACTLY the ingredient names provided
    2. Only include "grams" OR "ml" per ingredient (never both)
    3. Skip all optional fields like "notes", "state", etc.
    4. Use standard conversions:
       - 1 cup flour = 125g
       - 1 cup water = 240ml
       - 1 tbsp oil = 15ml
       - 1 tbsp butter = 14g

    Ingredients to convert:
    {json.dumps(ingredients, indent=2)}
    """
    
    try:
        response = gemini.generate_content(prompt)
        json_str = re.search(r'\{.*\}', response.text, re.DOTALL).group()
        result = json.loads(json_str)
        
        for item in result["results"]:
            item.pop("notes", None)
            item.pop("state", None)
            
            if "ml" in item and "grams" in item:
                if "flour" in item["ingredient"]:
                    del item["ml"]
                else:
                    del item["grams"]

        return result["results"]
    except Exception as e:
        print(f"Gemini Error: {str(e)}")
        return None

@app.get("/")
def home():
    return {"message": "API is live!", "endpoints": {"/convert": "POST"}}

@app.post("/convert")
async def convert_recipe(request: RecipeRequest):
    try:
        ingredients = extract_ingredients(request.text)
        if not ingredients:
            raise HTTPException(status_code=400, detail="No ingredients detected")
        
        result = convert_with_gemini(ingredients)
        if not result:
            raise HTTPException(status_code=500, detail="Conversion failed")
        
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start ngrok tunnel
public_url = ngrok.connect(8000)
print("Public URL:", public_url)

# Run FastAPI
uvicorn.run(app, host="0.0.0.0", port=8000)