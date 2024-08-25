const logo = document.getElementById('logo');
        const gameContainer = document.getElementById('game-container');
        const scoreDisplay = document.getElementById('score');
        let score = 0;

        function randomPosition() {
            const containerWidth = gameContainer.offsetWidth;
            const containerHeight = gameContainer.offsetHeight;
            const logoWidth = logo.offsetWidth;
            const logoHeight = logo.offsetHeight;

            // Generatng random positions within the container boundary
            const randomX = Math.floor(Math.random() * (containerWidth - logoWidth));
            const randomY = Math.floor(Math.random() * (containerHeight - logoHeight));

            // random positions to the logo
            logo.style.left = `${randomX}px`;
            logo.style.top = `${randomY}px`;
        }

        function randomMovement() {
            // Move the logo to a random position every 800 to 1500 milliseconds
            setInterval(() => {
                randomPosition();
            }, Math.floor(Math.random() * (1500 - 800 + 1)) + 800);
        }

        logo.addEventListener('click', () => {
            score++;
            scoreDisplay.textContent = `Score: ${score}`;
            randomPosition();  // change pos after being catched
        });

        // trigger the random movement when the page loads
        randomMovement();

        // Show notification when the page loads
        const notification = document.getElementById('notification');
        const closeBtn = document.getElementById('close-btn');

        function showNotification() {
            notification.classList.add('show');
            // Auto-hide notification after 5 seconds
            setTimeout(() => {
                notification.classList.remove('show');
            }, 5000);
        }

        closeBtn.addEventListener('click', () => {
            notification.classList.remove('show');
        });

        // Display the notification when the page loads
        window.onload = showNotification;