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
                a.href = directory.html_url;
                a.textContent = 'View Repository';
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
                listItem.classList.add('card-languages');
                const h3 = document.createElement('h3');
                h3.textContent = `${language}`;
                listItem.appendChild(h3);
                // listItem.innerHTML = `
                //     <h3>${language}</h3>
                //     <div class="progress-bar" style="width: ${percentage}%;"></div>
                // `;
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
        { question: 'What is Machine Learning?', answer: 'Machine Learning is a field of AI that enables computers to learn from data without being explicitly programmed.' },
        { question: 'Tell me about Machine Learning Repos.', answer: 'Machine Learning Repos is a curated collection of Machine Learning Repositories' },
        { question: 'How do I contribute to the repository?', answer: 'You can contribute by forking the repository, making changes, and submitting a pull request. Learn more here: <a href="https://github.com/recodehive/machine-learning-repos/blob/main/Website/README.md" target="_blank">here</a>' },
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

    const goToTopButton = document.getElementById('go-to-top');

    window.addEventListener('scroll', () => {
        if (window.scrollY > 300) {
            goToTopButton.style.display = 'block';
        } else {
            goToTopButton.style.display = 'none';
        }
    });

    goToTopButton.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    fetchDirectories();
    toggleLanguagesSection();
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
                            <a href="${contributor.html_url}" target="_blank" class="contributor-github">GitHub Profile</a>
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

