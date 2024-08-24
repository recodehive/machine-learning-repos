document.addEventListener('DOMContentLoaded', function() {
    const directoriesList = document.getElementById('directories');

    async function fetchDirectories() {
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

    fetchDirectories();
});