from django.http import HttpResponse, JsonResponse
import requests
from django.shortcuts import render
from datetime import datetime
from .prediction import predict_prices_for_future_random_forest, predict_prices_for_future_extra_trees

def home(request):
    # Choose the correct template to render
    return render(request, 'home.html')


def predict(request):
    if request.method == 'POST':
        print(request.POST)
        date = request.POST['date']
        print(date)
        predicted_price, predicted_high, predicted_low = predict_prices_for_future_random_forest(date)
        predicted_price = round(predicted_price, 2)
        predicted_high = round(predicted_high, 2)
        predicted_low = round(predicted_low, 2)
        return JsonResponse({
            'predicted_price': predicted_price,
            'predicted_high': predicted_high,
            'predicted_low': predicted_low
        })
    else:
        return HttpResponse(status=405)
    

def predict_ethereum_price(request):
    # Fetch current Ethereum price from an API
    response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd')
    if response.status_code == 200:
        actual_price = response.json()['ethereum']['usd']
    else:
        actual_price = None

    # Get the current date
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Call prediction functions for the current date
    rf_predicted_price, _, _ = predict_prices_for_future_random_forest(current_date)
    et_predicted_price, _, _ = predict_prices_for_future_extra_trees(current_date)

    # Pass predicted and actual prices to the template
    context = {
        'rf_predicted_price': rf_predicted_price,
        'et_predicted_price': et_predicted_price,
        'actual_price': actual_price,
    }

    return render(request, 'home.html', context)  
