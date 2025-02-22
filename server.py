from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/run-script', methods=['POST'])
def run_script():
    data = request.json
    input_path = data.get("input")
    output_path = data.get("output")

    if not input_path or not output_path:
        return jsonify({"error": "Missing input or output path"}), 400

    try:
        subprocess.run(["python", "toReel.py", "-i", input_path, "-o", output_path], check=True)

        return jsonify({"message": "Script executed successfully!", "output": output_path})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Script execution failed: {e}"}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)