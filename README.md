# CardioVascularDisease

Classification model with FastApi and Scikit Learn to Predict whether a patient have a the risk of a cardiovascular disease.


## App's architecture 
  
| data      
 &nbsp;&nbsp;&nbsp;&nbsp;| cardio_train.csv   
| fastapi   
 &nbsp;&nbsp;&nbsp;&nbsp;| main.py     
| model     
 &nbsp;&nbsp;&nbsp;&nbsp;| model.py        
| README.md      
| requirements.txt 


## How to deploy 
 - Clone our git repo (more details below) 
 - create or use your virtual environment set on python3
 - install the requirements.txt file
 - then on your terminal start your fast api app : uvicorn main:app --reload (make sure you are working in the right directory:fastapi)
 - on your local host : http://127.0.0.1:8000/docs you should be able to view all our app features

## Machine Learning 
 The ML model used to train and classify is chosen among other classifiers :
  - Nearest Neighbors
  - Decision Tree
  - Random Forest
  - AdaBoost
  - Naive Bayes
  - QDA
  
The algorithm only chooses the most accurate to load ad a pickle file



