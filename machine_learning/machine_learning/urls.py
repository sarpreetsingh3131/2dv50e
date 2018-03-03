from django.urls import path
from machine_learning.controller import controller

urlpatterns = [
    path('', controller.handle_request),
]
