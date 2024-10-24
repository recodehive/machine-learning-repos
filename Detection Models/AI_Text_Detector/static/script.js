function detectText() {
    document.getElementById('result').innerText = '';
    var textInput = document.getElementById('textInput');
    var fileInput = document.getElementById('fileInput');

    // Check if a file is uploaded
    if (fileInput.files.length > 0) {
        var file = fileInput.files[0];
        var reader = new FileReader();

        // Read the text content of the file
        reader.onload = function(event) {
            var text = event.target.result;

            // Send the text to the server for classification
            fetch('/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').innerText = 'Result: ' + data.result;
            })
            .catch(error => {
                console.error('Error:', error);
            });
        };

        reader.readAsText(file);
    } else {
        // If no file is uploaded, use the text from the textarea
        var text = textInput.value;

        // Send the text to the server for classification
        fetch('/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').innerText = 'Result: ' + data.result;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
}
