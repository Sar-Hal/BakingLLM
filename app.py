from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

# Ngrok URL for the backend API
API_URL = "https://85ee-34-169-58-207.ngrok-free.app/convert"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        recipe_text = request.form.get("recipe")
        if recipe_text:
            try:
                # Send the recipe text to the backend API
                response = requests.post(API_URL, json={"text": recipe_text})
                if response.status_code == 200:
                    result = response.json()
                    return render_template("index.html", result=result, recipe_text=recipe_text)
                else:
                    return render_template("index.html", error="Failed to process the recipe.", recipe_text=recipe_text)
            except Exception as e:
                return render_template("index.html", error=str(e), recipe_text=recipe_text)
        else:
            return render_template("index.html", error="Please enter a recipe.")

    # Render the page for GET requests
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
