from transformers import pipeline
from dotenv import load_dotenv
import google.generativeai as genai
import re
from fractions import Fraction
import os

# 1. Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 2. Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
client = genai.GenerativeModel("gemini-pro")

# 3. Improved ingredient extraction (NER-free version)
def extract_ingredients(text):
    ingredients = []
    pattern = r'-?\s*(\d+/\d+|\d+\.\d+|\d+\s\d+/\d+|\d+)\s*(cup|tbsp|tsp|oz|lb|teaspoon|tablespoon|pint|gallon|quart|ml|l)s?\s*(.*?)(?=\n|$)'
    
    for match in re.finditer(pattern, text, re.IGNORECASE):
        qty, unit, ingredient = match.groups()
        quantity = parse_quantity(qty)
        if quantity:
            # Clean ingredient name
            ingredient = re.sub(r'\(.*\)', '', ingredient).strip()
            ingredients.append({
                "ingredient": ingredient.lower(),
                "quantity": quantity,
                "unit": unit.lower()
            })
    return ingredients

def parse_quantity(qty_str):
    try:
        if ' ' in qty_str and '/' in qty_str:
            whole, fraction = qty_str.split()
            return float(whole) + float(Fraction(fraction))
        return float(Fraction(qty_str))
    except:
        return None

# 4. Gemini conversion with proper client
def convert_with_gemini(ingredients):
    system_prompt = """You are a professional baking converter. Convert these ingredients to exact gram measurements.
    Return ONLY JSON in this format:
    {
        "conversions": [
            {
                "ingredient": string,
                "original_amount": number,
                "original_unit": string,
                "grams": number,
                "notes": string
            }
        ]
    }
    Rules:
    1. For flour: 1 cup = 125g (all-purpose, spooned and leveled)
    2. For liquids: 1 cup = 240g (water), adjust density for others
    3. Include helpful notes about packing/measuring
    """
    
    user_prompt = json.dumps(ingredients, indent=2)
    
    try:
        response = client.generate_content(
            system_prompt + "\n\nConvert these:\n" + user_prompt
        )
        # Extract JSON from response
        result = json.loads(response.text.strip("` \n"))
        return result
    except Exception as e:
        print(f"Gemini Error: {str(e)}")
        return None

# 5. Main function
def convert_recipe_text(recipe_text):
    print("\n🔍 Extracting ingredients...")
    ingredients = extract_ingredients(recipe_text)
    
    if not ingredients:
        print("❌ No measurable ingredients found")
        return
    
    print("\n📋 Detected Ingredients:")
    for ing in ingredients:
        print(f"- {ing['quantity']} {ing['unit']} {ing['ingredient']}")
    
    print("\n⚡ Converting with Gemini...")
    result = convert_with_gemini(ingredients)
    
    if result:
        print("\n✅ Conversion Results:")
        for item in result["conversions"]:
            print(f"{item['original_amount']} {item['original_unit']} {item['ingredient']} → {item['grams']}g")
            print(f"   Note: {item['notes']}")
    else:
        print("❌ Conversion failed")

# Example usage
if __name__ == "__main__":
    test_recipe = """
    Classic Chocolate Chip Cookies:
    - 2 1/4 cups all-purpose flour
    - 1 tsp baking soda
    - 1 cup butter (softened)
    - 3/4 cup granulated sugar
    - 2 cups chocolate chips
    """
    
    convert_recipe_text(test_recipe)