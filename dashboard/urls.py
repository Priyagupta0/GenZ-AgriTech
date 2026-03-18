from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('disease-detection/', views.disease_detection, name='disease_detection'),
    path('weather-forecast/', views.weather_page, name='weather_page'),
    path('get_weather/', views.weather_forecast, name='weather_forecast'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('government-schemes/', views.government_schemes, name='government_schemes'),
    path('soil-detection/', views.soil_detection, name='soil_detection'),
    path('chat/api/', views.chat_api, name='chat_api'),
]
