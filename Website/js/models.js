// Function to fetch and display subdirectories (models) of the selected directory
async function fetchSubdirectories() {
    const params = new URLSearchParams(window.location.search);
    const directoryName = params.get('directory');
    const subdirectoriesList = document.getElementById('subdirectories');

    try {
        const response = await fetch(`https://api.github.com/repos/recodehive/machine-learning-repos/contents/${directoryName}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const subdirectories = data.filter(item => item.type === 'dir');

        subdirectories.forEach(subdirectory => {
            const li = document.createElement('li');
            li.classList.add('card');

            const h3 = document.createElement('h3');
            h3.textContent = subdirectory.name;

            const a = document.createElement('a');
            a.href = subdirectory.html_url;
            a.textContent = 'View Repository';
            a.classList.add('btn-view-repo');

            li.appendChild(h3);
            li.appendChild(a);
            subdirectoriesList.appendChild(li);
        });
    } catch (error) {
        console.error('Error fetching subdirectories:', error);
        subdirectoriesList.innerHTML = '<li class="card">Failed to load models.</li>';
    }
}

// Call the function when the page loads
document.addEventListener('DOMContentLoaded', fetchSubdirectories);
