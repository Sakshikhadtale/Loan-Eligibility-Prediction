from django.shortcuts import render

def about(request):
    context = {
        'page_title': 'About Us',
        'message': 'Welcome to our About page!'
    }
    return render(request, 'index.html', context)



















































































from django.views.decorators.csrf import csrf_exempt
result='yes'
@csrf_exempt
def loan_eligibility(request):
  if request.method == 'POST':
    loan_id = request.POST.get('Loan_Id')
    dependants = request.POST.get('Dependants')
    gender = request.POST.get('gender')
    self_employed = request.POST.get('Self Employed')
    married = request.POST.get('Married')
    highest_education = request.POST.get('Highest Education')
    applicant_income = int(request.POST.get('Applicant_Income'))
    coapplicant_income = request.POST.get('Coapplicant_Income')
    Credit_History = int(request.POST.get('Credit_History'))
    loan_amount = request.POST.get('Loan_Amount')
    
    # perform loan eligibility prediction based on user input data
    if applicant_income >= 50000 and Credit_History >= 700 :
        eligibility = 'Eligible'
        ans="y"
    elif applicant_income >= 35000 and Credit_History >= 600 :
        eligibility = 'Eligible with conditions'
        ans="y"
    elif applicant_income >= 25000 and Credit_History >= 500 :
        eligibility = 'Eligible with high interest rate'
        ans="y"
    else:
        eligibility = 'Not Eligible'
        ans='n'
    return render(request, 'index.html', {'result': eligibility,'ans':ans})
    
  else:
    return render(request, 'index.html')