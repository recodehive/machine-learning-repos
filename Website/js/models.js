// Function to fetch and display subdirectories (models) of the selected directory
async function fetchSubdirectories() {
    const params = new URLSearchParams(window.location.search);
    const directoryName = params.get('directory');
    const subdirectoriesList = document.getElementById('subdirectories');
    const loadingContainer = document.getElementById('loading-animation');

    // Show loading indicator
    loadingContainer.style.display = 'block';

    if (!directoryName) {
        subdirectoriesList.innerHTML = '<li class="card">No directory specified. Please try again.</li>';
        loadingContainer.style.display = 'none';
        return;
    }

    try {
        const response = await fetch(`https://api.github.com/repos/recodehive/machine-learning-repos/contents/${directoryName}`);
        
        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('Directory not found.');
            } else {
                throw new Error('Failed to fetch models. Please try again later.');
            }
        }

        const data = await response.json();
        const subdirectories = data.filter(item => item.type === 'dir');

        if (subdirectories.length === 0) {
            subdirectoriesList.innerHTML = '<li class="card">No models found in this directory.</li>';
        } else {
            // Build HTML for subdirectories
            let content = '';
            subdirectories.forEach(subdirectory => {
                content += `
                    <li class="card">
                        <h3>${subdirectory.name}</h3>
                        <a href="${subdirectory.html_url}" class="btn-view-repo" target="_blank">Explore the Repository</a>
                    </li>
                `;
            });
            subdirectoriesList.innerHTML = content;
        }

    } catch (error) {
        subdirectoriesList.innerHTML = `<li class="card">${error.message}</li>`;
    } finally {
        // Hide loading indicator
        loadingContainer.style.display = 'none';
    }
}

// Call the function when the page loads
document.addEventListener('DOMContentLoaded', fetchSubdirectories);
