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
            const directories = data.filter(item => item.type === 'dir' && item.name !== 'Website');

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
                listItem.innerHTML = `
                    <span>${language}</span>
                    <div class="progress-bar" style="width: ${percentage}%;"></div>
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

    fetchDirectories();
    toggleLanguagesSection();
    fetchRepoStats();
    toggleStatsSection();
});
