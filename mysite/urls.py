from django.urls import path
from . import views

urlpatterns = [
    path('', views.about, name='about'),
    path('loan-eligibility/', views.loan_eligibility, name='loan_eligibility'),
]