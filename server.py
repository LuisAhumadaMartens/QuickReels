from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import os
from pathlib import Path
import traceback
import logging
import cv2
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

def clean_upload_directory():
    """Remove all files from the upload directory."""
    for file in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {e}")

@app.route('/save-file', methods=['POST'])
def save_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Clean up the uploads directory before saving new file
        clean_upload_directory()

        # Reset progress.json
        with open('progress.json', 'w') as f:
            json.dump({
                'progress': 0,
                'status': 'Initializing...'
            }, f)

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

        def timestamp_to_seconds(timestamp):
            # Convert "MM:SS" format to seconds
            if not timestamp:
                return 0
            parts = timestamp.split(':')
            if len(parts) == 2:
                minutes, seconds = map(float, parts)
                return minutes * 60 + seconds
            return float(timestamp)

        # Get the directory where toReel.py is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        if output_type == 'single':
            output_path = os.path.join(UPLOAD_DIR, f"output.{input_path.split('.')[-1]}")
            cmd = ["python", "toReel.py", "-i", input_path, "-o", output_path]
        else:
            # Handle multiple crops with new argument format
            crops = data.get('crops', [])
            cmd = ["python", "toReel.py", "-i", input_path, "-mo"]
            
            for i, crop in enumerate(crops):
                output_name = f"output_{i+1}.{input_path.split('.')[-1]}"
                output_path = os.path.join(UPLOAD_DIR, output_name)
                
                # Convert timestamp to seconds, then to frames
                start_seconds = timestamp_to_seconds(crop['start'])
                end_seconds = timestamp_to_seconds(crop['end'])
                
                start_frame = int(start_seconds * fps)
                end_frame = int(end_seconds * fps)
                
                # Add output path and frame range for each crop
                cmd.extend([output_path, f"{start_frame}-{end_frame}"])

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
            output_files = [c[1] for c in zip(crops, [os.path.join(UPLOAD_DIR, f"output_{i+1}.{input_path.split('.')[-1]}") for i, _ in enumerate(crops)])]
            return jsonify({
                "message": "Success",
                "outputs": output_files
            })

    except Exception as e:
        logger.error(f"Error running script: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

# Add route to serve files from uploads directory
@app.route('/uploads/<path:filename>')
def serve_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# Add new endpoint to get progress
@app.route('/get-progress', methods=['GET'])
def get_progress():
    try:
        with open('progress.json', 'r') as f:
            progress_data = json.load(f)
        return jsonify(progress_data)
    except FileNotFoundError:
        return jsonify({'progress': 0, 'status': 'Initializing...'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)