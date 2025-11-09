# Aaroh_project/app.py
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from utils.llm_processor import get_aaroh_output # Will use this function soon!

# Load environment variables (must be the first step)
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)
# Add CORS support if you plan to connect a separate frontend (highly recommended)
# from flask_cors import CORS
# CORS(app)

# --- API Endpoint ---

@app.route('/simplify', methods=['POST'])
def simplify_text_endpoint():
    # 1. Input Validation
    if not request.is_json or 'text' not in request.json:
        return jsonify({"error": "Missing 'text' field in JSON request"}), 400

    complex_text = request.json.get('text')
    
    # 2. Call the AI Logic (will be implemented in utils/llm_processor.py)
    result, success = get_aaroh_output(complex_text)

    # 3. Handle Response
    if success:
        return jsonify(result), 200
    else:
        # If the LLM call fails, return a server error
        return jsonify({"error": result}), 500


# --- Run the Server ---
if __name__ == '__main__':
    # You must run this using `python app.py` while your (venv) is activeq
    app.run(debug=True)