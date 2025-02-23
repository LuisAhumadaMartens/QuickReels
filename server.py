from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import os
from pathlib import Path
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/')
def serve_html():
    return send_from_directory('.', 'index.html')

@app.route('/save-file', methods=['POST'])
def save_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save file
        filepath = os.path.join(UPLOAD_DIR, file.filename)
        file.save(filepath)
        logger.info(f"Saved file to: {filepath}")
        
        return jsonify({
            "message": "File saved successfully",
            "filepath": filepath
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        data = request.json
        input_path = data.get('input_path')
        output_path = os.path.join(UPLOAD_DIR, f"output.{input_path.split('.')[-1]}")

        # Get the directory where toReel.py is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Change to the correct directory
        os.chdir(script_dir)
        
        # Run the command directly without capturing output
        cmd = ["python", "toReel.py", "-i", input_path, "-o", output_path]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run without capturing output for maximum speed
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            raise Exception(f"Script failed with return code: {result.returncode}")

        return jsonify({
            "message": "Success",
            "output": output_path
        })

    except Exception as e:
        logger.error(f"Error running script: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)