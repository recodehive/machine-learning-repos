'use strict';

// Function to GET csrftoken from Cookie
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

var csrftoken = getCookie('csrftoken');

function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}

// Function to set Request Header with `CSRFTOKEN`
function setRequestHeader(){
    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });
}

// function updateValue(){
//     const selectedDate = document.getElementById('date').value;
//     setRequestHeader();

//     $.ajax({
//         dataType: 'json',
//         type: 'POST',
//         url: "/predict_ethereum_price/",
//         data: { date: selectedDate },  
//         success: function (data) {
//             console.log("Success:", data);
//             document.getElementById("result_model").innerHTML = "$" + data.predicted_price;
//             document.getElementById("result_high").innerHTML = "High: $" + data.predicted_high;             
//             document.getElementById("result_low").innerHTML = "Low: $" + data.predicted_low;
//         },
//         error: function (jqXHR) {
//             console.log("Error:", jqXHR);
//             alert("Update failed!");
//         }
//     });
// }


const exchangeRate = 84; 

function updateValue() {
    const selectedDate = document.getElementById('date').value;
    setRequestHeader();

    $.ajax({
        dataType: 'json',
        type: 'POST',
        url: "/predict_ethereum_price/",
        data: { date: selectedDate },
        success: function (data) {
            console.log("Success:", data);

            const predictedPriceDollars = data.predicted_price;
            const predictedHighDollars = data.predicted_high;
            const predictedLowDollars = data.predicted_low;

            const predictedPriceRupees = convertToRupees(predictedPriceDollars);
            const predictedHighRupees = convertToRupees(predictedHighDollars);
            const predictedLowRupees = convertToRupees(predictedLowDollars);

            document.getElementById("price_dollars").innerText = predictedPriceDollars;
            document.getElementById("price_rupees").innerText = predictedPriceRupees.toFixed(2);

            document.getElementById("high_dollars").innerText = predictedHighDollars;
            document.getElementById("high_rupees").innerText = predictedHighRupees.toFixed(2);

            document.getElementById("low_dollars").innerText = predictedLowDollars;
            document.getElementById("low_rupees").innerText = predictedLowRupees.toFixed(2);
        },
        error: function (jqXHR) {
            console.log("Error:", jqXHR);
            alert("Update failed!");
        }
    });
}

function convertToRupees(dollars) {
    return dollars * exchangeRate;
}



// Function to scroll to a specific section when a navbar link is clicked
function scrollToSection(sectionId) {
    document.getElementById(sectionId).scrollIntoView({
        behavior: 'smooth'
    });
}

// Add event listeners to navbar links
document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const sectionId = link.getAttribute('href').substring(1);
            scrollToSection(sectionId);
        });
    });
});

function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

window.addEventListener('load', function() {
    window.scrollTo(0, 0);
});


