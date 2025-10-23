document.getElementById('predictionForm').addEventListener('submit', function(event) {
    event.preventDefault();

    // Collect form data
    let formData = new FormData(this);

    // Convert form data to JSON object
    let user_input = {};
    formData.forEach(function(value, key){
        user_input[key] = value;
    });

    // Ensure numerical values are correctly handled
    user_input['Age'] = parseFloat(user_input['Age']);
    user_input['Number_of_Dependents'] = parseInt(user_input['Number_of_Dependents']);
    user_input['Work_Experience'] = parseFloat(user_input['Work_Experience']);
    user_input['Household_Size'] = parseInt(user_input['Household_Size']);

    // Send JSON object to server for prediction
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(user_input),
    })
    .then(response => response.json())
    .then(data => {
        // Display predicted income
        document.getElementById('result').textContent = 'Predicted Income: Rs ' + data.predicted_income.toFixed(2);
        document.getElementById('result').style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
