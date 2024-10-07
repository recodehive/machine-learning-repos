let model;

// Initialize the camera for video feed
async function initializeCamera() {
    const video = document.getElementById('video');
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
}

// Load the FaceMesh model (for client-side basic processing if needed)
async function initializeModel() {
    model = await facemesh.load();
    document.getElementById('status').textContent = 'Model loaded, ready for authentication...';
}

// Capture an image from the video feed
function captureImage() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/png');
}

// Register a new user with the captured image
async function register() {
    const username = document.getElementById('register-username').value;
    const image = captureImage();

    const response = await fetch('/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username, data: image })
    });

    const result = await response.json();
    document.getElementById('status').textContent = result.result || result.error;
}

// Login a user and authenticate their face
async function login() {
    const username = document.getElementById('login-username').value;
    const image = captureImage();

    const response = await fetch('/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username, data: image })
    });

    const result = await response.json();
    document.getElementById('status').textContent = result.result || result.error;

    if (result.result) {
        // Redirect to a dummy home page after successful login
        setTimeout(() => {
            window.location.href = "/static/home.html";
        }, 2000);
    }
}

// Toggle between register and login forms
function toggleAuth() {
    const registerContainer = document.getElementById('register-container');
    const loginContainer = document.getElementById('login-container');

    if (registerContainer.classList.contains('auth-hidden')) {
        registerContainer.classList.remove('auth-hidden');
        loginContainer.classList.add('auth-hidden');
    } else {
        registerContainer.classList.add('auth-hidden');
        loginContainer.classList.remove('auth-hidden');
    }
}

// Initialize everything on page load
initializeCamera();
initializeModel();
