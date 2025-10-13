"""
URL configuration for Stress_Prediction_Project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from . import view

urlpatterns = [
    path('', view.home, name='home'),
    path('signup/', view.signup, name = 'signup'),
    path('signin/', view.signin, name = 'signin'),
    path('signout/', view.signout, name = 'signout'),
    path('predict/', view.predict, name='predict'),
    path('activate/<uidb64>/<token>',view.activate, name ='activate')
]

