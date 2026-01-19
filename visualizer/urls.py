from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('kmeans/', views.kmeans_view, name='kmeans'),
    path('dbscan/', views.dbscan_view, name='dbscan'),
    path('naive-bayes/', views.naive_bayes_view, name='naive_bayes'),
]
