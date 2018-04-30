from django.urls import path
from machine_learner.controller import controller

urlpatterns = [
    path('', controller.training_testing),
]
