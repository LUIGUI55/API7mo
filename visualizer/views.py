from django.shortcuts import render
from .utils import generate_kmeans_graph, generate_dbscan_graph, generate_naive_bayes_graph, predict_spam

def home(request):
    return render(request, 'visualizer/home.html')

def kmeans_view(request):
    graph = generate_kmeans_graph()
    context = {'graph': graph, 'title': 'Clasificación K-Means', 'desc': 'K-Means aplicado a datos de transacciones reducidos por PCA para identificar grupos de fraude potencial.'}
    return render(request, 'visualizer/dashboard.html', context)

def dbscan_view(request):
    graph = generate_dbscan_graph()
    context = {'graph': graph, 'title': 'Clustering DBSCAN', 'desc': 'DBSCAN utilizado para la detección de anomalías, resaltando efectivamente el ruido que podría representar actividad fraudulenta.'}
    return render(request, 'visualizer/dashboard.html', context)

def naive_bayes_view(request):
    graph = generate_naive_bayes_graph()
    prediction = None
    
    if request.method == 'POST':
        text_to_check = request.POST.get('text_to_check')
        if text_to_check:
            prediction = predict_spam(text_to_check)
            
    context = {
        'graph': graph, 
        'title': 'Clasificación Naive Bayes', 
        'desc': 'Rendimiento del clasificador Naive Bayes (Matriz de Confusión) en la detección de correos SPAM.',
        'show_prediction': True,
        'prediction': prediction
    }
    return render(request, 'visualizer/dashboard.html', context)
