from django.urls import path,include
from . import views

app_name = 'mainapp'

urlpatterns = [
    path('',views.home,name='home'),
    path('predict_ethereum_price/',views.predict,name='predict'),
]