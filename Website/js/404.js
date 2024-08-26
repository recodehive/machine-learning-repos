const logo = document.getElementById('logo');
const gameContainer = document.getElementById('game-container');
const scoreDisplay = document.getElementById('score');
let score = 0;
let intervalId;

function randomPosition() {
    const containerWidth = gameContainer.offsetWidth;
    const containerHeight = gameContainer.offsetHeight;
    const logoWidth = logo.offsetWidth;
    const logoHeight = logo.offsetHeight;

    const randomX = Math.floor(Math.random() * (containerWidth - logoWidth));
    const randomY = Math.floor(Math.random() * (containerHeight - logoHeight));

    logo.style.left = `${randomX}px`;
    logo.style.top = `${randomY}px`;
}

function randomMovement() {
    const minInterval = 990;
    const maxInterval = 1500;

    intervalId = setInterval(() => {
        randomPosition();
    }, Math.floor(Math.random() * (maxInterval - minInterval + 1)) + minInterval);
}

logo.addEventListener('click', () => {
    score++;
    scoreDisplay.textContent = `Score: ${score}`;
    randomPosition();
    increaseDifficulty();
    playSound('click');
});

function increaseDifficulty() {
    clearInterval(intervalId);
    const newInterval = Math.max(500, 1500 - score * 10);
    randomMovement();
}

function playSound(type) {
    const audio = new Audio(`/sounds/${type}.mp3`);
    audio.play();
}

function showNotification() {
    const notification = document.getElementById('notification');
    notification.classList.add('show');
    setTimeout(() => {
        notification.classList.remove('show');
    }, 5000);
}

const closeBtn = document.getElementById('close-btn');
closeBtn.addEventListener('click', () => {
    const notification = document.getElementById('notification');
    notification.classList.remove('show');
});

window.onload = () => {
    showNotification();
    randomMovement();
};
