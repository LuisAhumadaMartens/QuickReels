from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
from pathlib import Path
import traceback
import logging
import tempfile
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        # Log request details
        logger.debug("Received request")
        logger.debug(f"Request files: {request.files}")
        logger.debug(f"Request form: {request.form}")
        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Python executable: {sys.executable}")
        
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({"error": "No file provided"}), 400
        
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            logger.error("Empty filename")
            return jsonify({"error": "No file selected"}), 400

        # Create a temporary directory to store the uploaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file
            input_path = os.path.join(temp_dir, uploaded_file.filename)
            uploaded_file.save(input_path)
            
            logger.debug(f"Saved uploaded file to: {input_path}")
            logger.debug(f"File exists: {os.path.exists(input_path)}")
            logger.debug(f"File size: {os.path.getsize(input_path)}")

            # Set output path in Downloads folder
            downloads_path = str(Path.home() / "Downloads")
            output_filename = f"output.{uploaded_file.filename.split('.')[-1]}"
            output_path = os.path.join(downloads_path, output_filename)
            
            logger.debug(f"Output will be saved to: {output_path}")
            logger.debug(f"Downloads path exists: {os.path.exists(downloads_path)}")

            # Check if toReel.py exists
            toreel_path = os.path.join(os.getcwd(), "toReel.py")
            logger.debug(f"toReel.py path: {toreel_path}")
            logger.debug(f"toReel.py exists: {os.path.exists(toreel_path)}")

            # Run toReel.py with the provided paths
            cmd = ["python", toreel_path, "-i", input_path, "-o", output_path]
            logger.debug(f"Executing command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Log the command output
            logger.debug(f"Command stdout: {result.stdout}")
            logger.debug(f"Command stderr: {result.stderr}")

            return jsonify({
                "message": "Script executed successfully!",
                "output": output_path,
                "stdout": result.stdout,
                "stderr": result.stderr
            })

    except subprocess.CalledProcessError as e:
        error_msg = f"Script execution failed:\nCommand: {e.cmd}\nExit code: {e.returncode}\nOutput: {e.output}\nError: {e.stderr}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)