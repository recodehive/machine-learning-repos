from flask import Flask, send_from_directory, request, jsonify
import requests

app = Flask(__name__)

TEE_SERVER_URL = 'http://localhost:6000/secure'

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/register', methods=['POST'])
def register():
    try:
        payload = {
            'action': 'register',
            'username': request.json['username'],
            'data': request.json['data']  # Base64 encoded image
        }

        response = requests.post(TEE_SERVER_URL, json=payload)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to communicate with TEE server"}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        payload = {
            'action': 'login',
            'username': request.json['username'],
            'data': request.json['data']  # Base64 encoded image
        }

        response = requests.post(TEE_SERVER_URL, json=payload)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to communicate with TEE server"}), 500
    
@app.route('/home')
def home():
    return send_from_directory('static', 'home.html') 

if __name__ == '__main__':
    app.run(port=5000)
