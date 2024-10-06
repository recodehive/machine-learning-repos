from flask import Flask, request, jsonify
import os
import base64

app = Flask(__name__)

# Directory where faces will be stored
FACE_STORAGE_DIR = 'faces'

# Ensure the directory exists
if not os.path.exists(FACE_STORAGE_DIR):
    os.makedirs(FACE_STORAGE_DIR)

def save_face(username, face_data):
    face_path = os.path.join(FACE_STORAGE_DIR, f'{username}.png')
    with open(face_path, 'wb') as f:
        f.write(base64.b64decode(face_data.split(',')[1]))

def face_exists(username):
    return os.path.exists(os.path.join(FACE_STORAGE_DIR, f'{username}.png'))

@app.route('/secure', methods=['POST'])
def secure():
    action = request.json.get('action')
    username = request.json.get('username')
    face_data = request.json.get('data')

    if action == 'register':
        # Save the face image
        save_face(username, face_data)
        return jsonify({'result': f'User {username} registered successfully.'})

    elif action == 'login':
        if not face_exists(username):
            return jsonify({'error': 'Username not found'}), 404

        # For simplicity, simulate face liveness check
        if "data:image/png" in face_data:
            return jsonify({'result': 'Face authenticated successfully.'})
        else:
            return jsonify({'error': 'Liveness check failed'}), 400

    return jsonify({'error': 'Invalid action'}), 400

if __name__ == '__main__':
    app.run(port=6000)
