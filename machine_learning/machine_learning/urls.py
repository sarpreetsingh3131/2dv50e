from django.urls import path
from machine_learning.controller import controller

urlpatterns = [
    path('', controller.online_training_testing),
]
