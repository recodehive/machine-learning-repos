document.addEventListener('DOMContentLoaded', function() {
    // Function to fetch and display directories
    async function fetchDirectories() {
        const directoriesList = document.getElementById('directories');
        try {
            const response = await fetch('/api/github/repos');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            const directories = data.filter(item => item.type === 'dir' && item.name !== 'Website' && item.name !== '.github');
    
            directories.forEach(directory => {
                const li = document.createElement('li');
                li.classList.add('card');
    
                const h3 = document.createElement('h3');
                h3.textContent = directory.name;
    
                const a = document.createElement('a');
                a.href = `models.html?directory=${encodeURIComponent(directory.name)}`;
                a.textContent = 'View Models';
                a.classList.add('btn-view-repo');
    
                li.appendChild(h3);
                li.appendChild(a);
                directoriesList.appendChild(li);
            });
        } catch (error) {
            console.error('Error fetching directories:', error);
            directoriesList.innerHTML = '<li class="card">Failed to load directories.</li>';
        }
    }
    
    // Function to fetch and count subdirectories for each directory
    async function fetchSubdirectoryCounts() {
        try {
            const directoriesResponse = await fetch('/api/github/repos');
            if (!directoriesResponse.ok) {
                throw new Error(`HTTP error! status: ${directoriesResponse.status}`);
            }
            const directoriesData = await directoriesResponse.json();
            const directories = directoriesData.filter(item => item.type === 'dir' && item.name !== 'Website');
            const directoryCounts = [];
    
            for (const directory of directories) {
                const subResponse = await fetch(`https://api.github.com/repos/recodehive/machine-learning-repos/contents/${directory.name}`);
                if (!subResponse.ok) {
                    throw new Error(`HTTP error! status: ${subResponse.status} for ${directory.name}`);
                }
                const subData = await subResponse.json();
                const subDirectoriesCount = subData.filter(item => item.type === 'dir').length;
                directoryCounts.push({ name: directory.name, count: subDirectoriesCount });
            }
    
            return directoryCounts;
        } catch (error) {
            console.error('Error fetching subdirectory counts:', error);
        }
    }
    
        // Function to toggle languages section
        function toggleLanguagesSection() {
            const toggleLanguagesButton = document.createElement('button');
            toggleLanguagesButton.id = 'toggle-languages';
            toggleLanguagesButton.textContent = 'Show Languages Used';
            const languagesList = document.getElementById('language-list');
            document.getElementById('languages').insertBefore(toggleLanguagesButton, languagesList);
    
            languagesList.style.display = 'none';
            
            toggleLanguagesButton.addEventListener('click', function() {
                languagesList.style.display = languagesList.style.display === 'none' ? 'block' : 'none';
                toggleLanguagesButton.textContent = languagesList.style.display === 'none' ? 'Show Languages Used' : 'Hide Languages Used';
            });
        }
    
        // Function to fetch and display repository stats
        async function fetchRepoStats() {
            const repoOwner = 'recodehive';
            const repoName = 'machine-learning-repos';
            const apiUrl = `https://api.github.com/repos/${repoOwner}/${repoName}`;
    
            try {
                // Fetch repository data
                const repoResponse = await fetch(apiUrl);
                const repoData = await repoResponse.json();
    
                // Populate statistics cards
                document.getElementById('total-stars').innerText = repoData.stargazers_count;
                document.getElementById('total-forks').innerText = repoData.forks_count;
                document.getElementById('open-issues').innerText = repoData.open_issues_count;
                document.getElementById('license').innerText = repoData.license ? repoData.license.spdx_id : 'No License';
                document.getElementById('repo-size').innerText = (repoData.size / 1024).toFixed(2) + ' MB';
    
                // Fetch and display languages
                const languagesResponse = await fetch(`${apiUrl}/languages`);
                const languagesData = await languagesResponse.json();
                const languageList = document.getElementById('language-list');
                const totalBytes = Object.values(languagesData).reduce((acc, val) => acc + val, 0);
                let mostUsedLanguage = { name: '', bytes: 0 };
    
                for (const [language, bytes] of Object.entries(languagesData)) {
                    const percentage = ((bytes / totalBytes) * 100).toFixed(2);
                    const listItem = document.createElement('li');
                    listItem.innerHTML = `
                        <span>${language}</span>
                        <div class="progress-bar-container">
                        <div class="progress-bar" style="width: ${percentage}%;">${percentage}%</div>
                        </div>
                    `;
                    languageList.appendChild(listItem);
    
                    if (bytes > mostUsedLanguage.bytes) {
                        mostUsedLanguage = { name: language, bytes: bytes };
                    }
                }
    
                document.getElementById('most-used-language').innerText = mostUsedLanguage.name;
    
            } catch (error) {
                console.error('Error fetching data from GitHub API:', error);
            }
        }
        async function createPieChart() {
            const directoryCounts = await fetchSubdirectoryCounts();
            const ctx = document.getElementById('milestoneChart').getContext('2d');
            const labels = directoryCounts.map(dir => dir.name);
            const data = directoryCounts.map(dir => dir.count);
            
            const chart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: [
                            '#FF6384',
                            '#36A2EB',
                            '#FFCE56',
                            '#4BC0C0',
                            '#9966FF',
                            '#FF9F40',
                            '#01A02A',
                            '#FA5F20'
                        ],
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            display: false // Disable Chart.js default legend
                        }
                    }
                }
            });
        
            // Inject custom legend
            const legendContainer = document.querySelector('.legend');
            legendContainer.innerHTML = labels.map((label, index) => `
                <span style="color: ${chart.data.datasets[0].backgroundColor[index]}">${label}</span>
            `).join('');
        }
        
        // Function to toggle statistics section
        function toggleStatsSection() {
            const toggleButton = document.getElementById('toggle-stats');
            const statsSection = document.getElementById('statistics-cards');
            const toggleTextDisplay = document.getElementById('display');
            statsSection.style.display = 'none';
    
            toggleButton.addEventListener('click', function() {
                statsSection.style.display = statsSection.style.display === 'none' ? 'block' : 'none';
                toggleTextDisplay.textContent = statsSection.style.display === 'none' ? 'Show' : 'Hide';
            });
        }
    
        const dropdownButton = document.getElementById('dropdownButton');
        const dropdownMenu = document.getElementById('dropdownMenu');
    
        dropdownButton.addEventListener('click', function(event) {
            dropdownMenu.classList.toggle('show');
            event.stopPropagation(); // Prevent click event from bubbling up
        });
    
        window.addEventListener('click', function(event) {
            if (!dropdownButton.contains(event.target)) {
                dropdownMenu.classList.remove('show');
            }
        });
    
        const chatbotButton = document.getElementById('chatbot-button');
        const chatbot = document.getElementById('chatbot');
        const closeChatbot = document.getElementById('close-chatbot');
        const messagesContainer = document.getElementById('chatbot-messages');
        const inputField = document.getElementById('chatbot-input');
        const sendButton = document.getElementById('chatbot-send');
        const questionList = document.getElementById('question-list');
        let questionsRendered = false;
    
        const messages = [
            { text: 'Hello! Welcome to Machine Learning Repos', type: 'bot' }
        ];
    
        // hardcoded questions and answers      
          const questionsAndAnswers = [
                { question: 'What is RecoderHive?',answer: 'RecodeHive is a community-driven platform offering curated machine learning repositories'},
                { question: 'What is Machine Learning?', answer: 'Machine Learning is a field of AI that enables computers to learn from data without being explicitly programmed.' },
                { question: 'Tell me about Machine Learning Repos.', answer: 'Machine Learning Repos is a curated collection of Machine Learning Repositories' },
                { question: 'How do I contribute to the repository?', answer: 'You can contribute by forking the repository, making changes, and submitting a pull request. Learn more <a href="https://github.com/recodehive/machine-learning-repos/blob/main/Website/README.md" target="_blank">here</a>' },
                { question: 'How many repositories are included in this collection?', answer: 'There are multiple repositories included, each covering various aspects of Machine Learning.' },
                { question: 'What are the main topics covered by these repositories?', answer: 'The repositories cover topics like data preprocessing, model training, NLP, and more.' },
                { question: 'Does the repository offer any courses?', answer: 'Yes, the repository provides links to courses related to Machine Learning.' },
                { question: 'What programming languages are used in these repositories?', answer: 'The repositories utilize languages such as Python, R, HTML, CSS, JavaScript and others.' },
                { question: 'Which frameworks are utilized in these repositories?', answer: 'Frameworks like TensorFlow, PyTorch, and Scikit-Learn are used.' },
                { question: 'What is the most popular repository in the collection?', answer: 'The most popular repository is the "Awesome Machine Learning" collection.' },
                { question: 'Are there any projects focusing on NLP in this collection?', answer: 'Yes, there are projects specifically focused on Natural Language Processing (NLP).' },
                { question: 'How many topics are covered in the repository?', answer: 'The repository covers several key topics, including data science, deep learning, and more.' },
                { question: 'Does the repository provide any tutorials?', answer: 'Yes, there are tutorials available that help users understand various machine learning concepts.' },
                { question: 'What is the purpose of the repository?', answer: 'The repository aims to provide a comprehensive collection of resources and projects for learning and applying machine learning.' },
                { question: 'Are there any datasets included in the repository?', answer: 'Yes, some repositories include datasets that can be used for training and testing machine learning models.' },
                { question: 'How frequently is the repository updated?', answer: 'The repository is regularly updated with new content and improvements.' }
            ];
            
        
    
        function renderMessages() {
            messagesContainer.innerHTML = '';
            messages.forEach((msg) => {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${msg.type}`;
                messageDiv.innerHTML = `<span class="message-text">${msg.text}</span>`;
                messagesContainer.appendChild(messageDiv);
            });
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    
        function addMessage(text, type) {
            messages.push({ text, type });
            renderMessages();
        }
    
        function renderQuestions() {
        questionsAndAnswers.forEach((qa, index) => {
            const listItem = document.createElement('li');
            listItem.textContent = qa.question;
            listItem.addEventListener('click', () => {
                addMessage(qa.question, 'user');
                setTimeout(() => addMessage(qa.answer, 'bot'), 500);
            });
            questionList.appendChild(listItem);
        });
        }
    
        chatbotButton.addEventListener('click', () => {
            chatbot.classList.add('active');
            renderMessages();
            if(questionsRendered===false){
                questionsRendered=true;
                renderQuestions();
            }
        });
    
        closeChatbot.addEventListener('click', () => {
            chatbot.classList.remove('active');
        });
    
        sendButton.addEventListener('click', () => {
            const userInput = inputField.value.trim();
            if (userInput) {
                addMessage(userInput, 'user');
                inputField.value = '';
                setTimeout(() => addMessage('Choose from the list of questions!', 'bot'), 500);    // response by bot
            }
        });
    
        inputField.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendButton.click();
            }
        });
    
       // Get the button
    const scrollTopBtn = document.getElementById("scroll-top-btn");
    
    // Show the button when the user scrolls down 100px from the top of the document
    window.onscroll = function() {
        if (document.body.scrollTop > 100 || document.documentElement.scrollTop > 100) {
            scrollTopBtn.style.display = "block";
        } else {
            scrollTopBtn.style.display = "none";
        }
    };
    
    // When the user clicks the button, scroll to the top of the document
    scrollTopBtn.addEventListener("click", function() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    
        fetchDirectories();
        createPieChart();
        fetchRepoStats();
        toggleStatsSection();
    });
    
    document.addEventListener("DOMContentLoaded", function() {
        fetchContributors();
    
        function fetchContributors() {
            const repoOwner = 'recodehive'; // Replace with your repository owner
            const repoName = 'machine-learning-repos'; // Replace with your repository name
            const apiUrl = `https://api.github.com/repos/${repoOwner}/${repoName}/contributors`;
    
            fetch(apiUrl)
                .then(response => response.json())
                .then(contributors => {
                    const contributorsGrid = document.getElementById('contributors-grid');
                    
                    contributors.forEach(contributor => {
                        const contributorDiv = document.createElement('div');
                        contributorDiv.className = 'contributor';
    
                        contributorDiv.innerHTML = `
                            <img src="${contributor.avatar_url}" alt="${contributor.login}" class="contributor-image">
                            <div class="contributor-info">
                                <a href="${contributor.html_url}" target="_blank" class="contributor-name">${contributor.login}</a>
                            </div>
                        `;
    
                        contributorsGrid.appendChild(contributorDiv);
                    });
                })
                .catch(error => {
                    console.error('Error fetching contributors:', error);
                });
        }
    });
    
    const toggleDarkModeButton = document.getElementById('toggle-dark-mode');
    const body = document.body;
    
    // function to apply the theme based on the stored value
    function applyTheme(theme) {
        // Remove all theme classes
        body.classList.remove('light-mode', 'dark-mode', 'blue-mode');
    
        // Apply the new theme class
        body.classList.add(theme);
    
        // Update the icon based on the theme
        const icon = toggleDarkModeButton.querySelector('i');
        if (theme === 'dark-mode') {
            icon.classList.add('fa-adjust');
            icon.classList.remove('fa-moon', 'fa-sun');
        } else if (theme === 'light-mode') {
            icon.classList.add('fa-sun');
            icon.classList.remove('fa-moon', 'fa-adjust');
        } else if (theme === 'blue-mode') {
            icon.classList.add('fa-moon');
            icon.classList.remove('fa-sun', 'fa-adjust');
        }
    
        // Save the current theme in localStorage
        localStorage.setItem('theme', theme);
    }
    
    // Check for the saved theme in localStorage
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        applyTheme(savedTheme);
    } else {
        // Set default theme to light
        applyTheme('light-mode');
    }
    
    let themes = {
        'light': 'light-mode',
        'dark': 'dark-mode',
        'blue': 'blue-mode'
    };
    
    let i = 0; // For light-mode
    
    function toggleTheme() {
        i++;
        if (i >= Object.keys(themes).length) {
            i = 0;
        }
    
        const newTheme = Object.keys(themes)[i];
        console.log(newTheme);
        applyTheme(themes[newTheme]);
    }
    
    toggleDarkModeButton.addEventListener('click', toggleTheme);
    
    document.addEventListener('keydown', function (event) {
        // Check if the 'Q' key is pressed along with the 'Ctrl' key
        if (event.ctrlKey && (event.key === 'q' || event.key === 'Q')) {
            event.preventDefault(); // Prevent the default action (if any)
            toggleTheme(); // Call the function to toggle the theme
        }
    });
    
    
    function hamburger() {
        const line = document.getElementById("line");
        const navLinks = document.querySelector(".nav-links");
    
        line.classList.toggle("change");
        navLinks.classList.toggle("active");
    }
    
    document.getElementById("line").addEventListener("click", hamburger);