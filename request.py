import requests

url= 'http://localhost:5000/predict_api'
r=requests.post(url,json={'age':37,	'sex':1,'cp':3,'trestbps':130,'chol':250,'fbs':0,'restecg':0,'thalach':187,'exang':0,'oldpeak':3.5,'slope':3,'ca':0,'thal':3})

print(r.json())
