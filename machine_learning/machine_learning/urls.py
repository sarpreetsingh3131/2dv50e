from django.urls import path
from machine_learning.controller import controller

urlpatterns = [
    path('', controller.online_training_testing),
    path('offline_training', controller.offline_training),

]
