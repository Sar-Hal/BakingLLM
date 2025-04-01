from transformers import pipeline
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
from fractions import Fraction
import json

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize models
food_ner = pipeline(
    "token-classification",
    model="Dizex/InstaFoodRoBERTa-NER",
    aggregation_strategy="simple",
    device=-1
)
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")

def parse_quantity(qty_str):
    try:
        if ' ' in qty_str and '/' in qty_str:
            whole, fraction = qty_str.split()
            return float(whole) + float(Fraction(fraction))
        return float(Fraction(qty_str))
    except:
        return None

def extract_ingredients(text):
    # Updated unit pattern with plural support
    unit_pattern = r'\b(cups?|tbsps?|tsps?|oz|lbs?|teaspoons?|tablespoons?|g|kg|ml)\b'
    ingredients = []

    # Modified regex to handle leading hyphens and spaces
    for match in re.finditer(
        r'-\s*((\d+/\d+|\d+\.\d+|\d+\s\d+/\d+|\d+)\s*([a-zA-Z]+)\s+.*?)(?=,|\n|$)',
        text,
        re.IGNORECASE
    ):
        full_match = match.group(1)
        parts = [p.strip() for p in re.split(r',\s*(?=\d)', full_match)]

        for part in parts:
            # Enhanced regex with unit normalization
            match = re.search(
                r'^(\d+/\d+|\d+\.\d+|\d+\s\d+/\d+|\d+)\s*([a-zA-Z]+)\s*(.*)',
                part,
                re.IGNORECASE
            )
            if not match:
                continue

            qty, unit, ingredient_part = match.groups()
            
            # Normalize plural units to singular
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
    Convert these baking ingredients to exact grams or milliletres. Return ONLY JSON:

    Rules:
    - Use professional baking standards.
    - For dry ingredients (flour, sugar), specify if packed/loose.
    - Return format:
    {{
      "ingredient": string,
      "grams": number, (if dry)
      "ml": number, (if wet)
    }}

    Input to convert:
    {json.dumps(ingredients, indent=2)}
    """
    
    try:
        response = gemini.generate_content(prompt)
        clean_response = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_response)
    except Exception as e:
        print(f"Gemini Error: {str(e)}")
        return None

def process_recipe(text):
    print("🔍 Extracting ingredients...")
    ingredients = extract_ingredients(text)
    
    if not ingredients:
        print("❌ No ingredients detected in:", text)
        return
    
    print("\n📋 Detected Ingredients:")
    for ing in ingredients:
        print(f"- {ing['quantity']} {ing['unit']} {ing['ingredient']}")
    
    print("\n⚡ Converting with Gemini...")
    result = convert_with_gemini(ingredients)
    
    if result:
        print("\n✅ Precision Conversions:")
        for item in result:
            print(f"{item['ingredient']}: {item['grams']}g - {item.get('notes', '')}")

# Test with your recipe
if __name__ == "__main__":
    recipe =  """
    - 2 1/4 cups all-purpose flour
    - 1 teaspoon baking soda
    - 1 cup unsalted butter
    - 3/4 cup packed brown sugar
    - 20 cups chocolate chips
    """
    process_recipe(recipe)