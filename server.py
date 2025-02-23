from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import os
from pathlib import Path
import traceback
import logging
import cv2

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
        output_type = data.get('output_type', 'single')  # 'single' or 'multiple'
        
        # Get video metadata to calculate frames from timestamps
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Get the directory where toReel.py is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        if output_type == 'single':
            output_path = os.path.join(UPLOAD_DIR, f"output.{input_path.split('.')[-1]}")
            cmd = ["python", "toReel.py", "-i", input_path, "-o", output_path]
        else:
            # Handle multiple crops
            crops = data.get('crops', [])
            cmd = ["python", "toReel.py", "-i", input_path]
            
            for crop in crops:
                output_name = f"output_{crop['start']}_{crop['end']}.{input_path.split('.')[-1]}"
                output_path = os.path.join(UPLOAD_DIR, output_name)
                
                # Convert time (in seconds) to frames
                start_frame = int(float(crop['start']) * fps)
                end_frame = int(float(crop['end']) * fps)
                
                cmd.extend(["-c", output_path, f"{start_frame}-{end_frame}"])

        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            raise Exception(f"Script failed with return code: {result.returncode}")

        # Return appropriate response based on output type
        if output_type == 'single':
            return jsonify({
                "message": "Success",
                "output": output_path
            })
        else:
            # For multiple crops, return all output paths
            output_files = [c[1] for c in zip(crops, [os.path.join(UPLOAD_DIR, f"output_{c['start']}_{c['end']}.{input_path.split('.')[-1]}") for c in crops])]
            return jsonify({
                "message": "Success",
                "outputs": output_files
            })

    except Exception as e:
        logger.error(f"Error running script: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)