from django.shortcuts import render

def home(request):
    context = {'message': 'Welcome to my website!'}
    return render(request, 'index.html', context)

# def blog(request):
#     posts = "post"
#     context = {'posts': posts}
#     return render(request, 'index.html', context)