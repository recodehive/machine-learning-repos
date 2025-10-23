
var strings = [
    'Welcome to our DeepFake Detection Website.',
    'Our advanced AI algorithms analyze images to detect any signs of manipulations.',
    'Our mission is to provide a reliable and efficient solution to combat the spread of deepfakes.'
];

var typingContainer = document.querySelector('.typing');
function typeString(string) {
    return new Promise(resolve => {
        var i = 0;
        var typingInterval = setInterval(function () {
            typingContainer.textContent += string[i];
            i++;
            if (i === string.length) {
                clearInterval(typingInterval); 
                typingContainer.innerHTML += '<br>'; 
                resolve(); 
            }
        }, 50); 
    });
}

async function typeAllStrings() {
    for (const string of strings) {
        await typeString(string); 
    }
}

typeAllStrings();
