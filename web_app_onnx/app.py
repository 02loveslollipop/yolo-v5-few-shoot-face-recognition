import os
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file

app = Flask(__name__, static_folder='.', static_url_path='')
BASE_DIR = Path(__file__).resolve().parent

# Use the environment variable for Heroku, or a default for local testing
API_TOKEN = os.environ.get("API_TOKEN", "EWCPGfYmPP0Qlu/80cBJg3PEEPMAmJWdYxpmsLMyqngNvITGSsTV2X+weI9j1pyRBRpCc6BnIXg1NInHhVINhXUWiqa4+SohE8VppvHUNPohr2yPlUdQws6AGB/6qhwexCYYx+ld/PTkdH2B9w1aTXQZMH48bCfjKE4U1wPPGdI=")

@app.route('/')
def index():
    return send_file(BASE_DIR / 'index.html')

@app.route('/app.js')
def serve_js():
    return send_file(BASE_DIR / 'app.js')

@app.route('/api/model/<path:filename>')
def serve_model(filename):
    """
    Secure endpoint to serve model weights. 
    Requires an Authorization header with the correct Bearer token.
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f"Bearer {API_TOKEN}":
        return jsonify({"error": "Unauthorized user. Invalid or missing API token."}), 401
    
    # Securely serve the file from the 'model' directory
    return send_from_directory(BASE_DIR / 'model', filename)

if __name__ == '__main__':
    # Heroku dynamically maps the PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
