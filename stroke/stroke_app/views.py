from django.shortcuts import render
from stroke_app.models import users

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# Create your views here.


def loginview(request):
    return render(request,'login.html')

def registrationview(request):
    return  render(request,'registration.html')

def saveuserview(request):
    username=request.POST['username']
    password = request.POST['password']
    name = request.POST['name']
    email = request.POST['email']
    phone = request.POST['phone']
    address = request.POST['address']

    newusers=users(username=username,password=password,name=name,email=email,phone=phone,address=address)
    newusers.save()
    return render(request, 'login.html')

def verifyuserview(request):
    username=request.POST["username"]
    password=request.POST["password"]

    user= users.objects.filter(username=username)

    for u in user:
        if u.password==password:
            return render(request,'home.html')
        else:
            return render(request, 'login.html')

def homeview(request):
    return render(request,'home.html')

def resultview(request):

    age=request.POST['age']
    hypertension=request.POST['bp']
    heart_disease=request.POST['hd']
    avg_glucose_level=request.POST['glucose']
    bmi=request.POST['bmi']

    if request.POST['gender']=='Male':
        gender=1
    else:
        gender=0

    if request.POST['married']=='Yes':
        ever_married=1
    else:
        ever_married=0

    jobs=['Private','Self-employed','Govt_job','children']
    for job in jobs:
        if request.POST['work'] == job:
            work_type = jobs.index(job)

    if request.POST['residence']=='Urban':
        Residence_type=1
    else:
        Residence_type=0

    smoke=['formerly smoked','never smoked','smokes','Unknown']
    for item in smoke:
        if request.POST['smoke']==item:
            smoking_status=smoke.index(item)

    #ML code
    file = pd.read_csv('brain_stroke.csv')

    encoder = preprocessing.LabelEncoder()
    file['gender'] = encoder.fit_transform(file['gender'])
    file['ever_married'] = encoder.fit_transform(file['ever_married'])
    file['work_type'] = encoder.fit_transform(file['work_type'])
    file['Residence_type'] = encoder.fit_transform(file['Residence_type'])
    file['smoking_status'] = encoder.fit_transform(file['smoking_status'])

    x = file.iloc[:, :-1]
    y = file.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)
    model = RandomForestClassifier(random_state=48)
    model.fit(x_train.values, y_train.values)

    # values1=[1,67.0,0,1,1,1,1,228.69,36.6,1]
    # values0 = [1, 41.0, 0, 0, 0, 1, 0, 70.15, 29.8, 1]
    values=[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]
    values = np.reshape(values, (1, -1))
    predicted_result = model.predict(values)

    if predicted_result[0] == 0:
        result="You are normal"

    else:
        result="You have possibility of stroke"

    return render(request, 'result.html',{'result':result})


