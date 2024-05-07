# import libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import seaborn as sns
import os 
# add some text
root = os.getcwd()
img_path = os.path.join(root,"ml_photo.jpg")
st.title("Machine Learning Algorithms")
st.image(img_path)  

st.write(''''
# Explore different classifier 
which one is the best?         
         ''')
# side bar selectboxes

dataset_name = st.sidebar.selectbox("Select Dataset",("breast-cancer","diabetes","iris"))
classifier_name = st.sidebar.selectbox("Select Classifier",("SVM","KNN","Random Forest"))

def load_dataset(dataset_name):
    if dataset_name == "breast-cancer":
        data = datasets.load_breast_cancer()    
    elif dataset_name == "diabetes":
        data = datasets.load_diabetes()
    else:
        data = datasets.load_iris()
        
    x = data.data
    y = data.target
    return x ,y
x,y = load_dataset(dataset_name)
st.write(dataset_name," ","dataset shape is" ,x.shape)
st.write("Number of classes is : ",len(np.unique(y)))

        
def add_parameter(clf_name):
    params = {}
    if classifier_name == "KNN":
        k = st.sidebar.slider("k",1,15)
        params["k"] = k
    elif classifier_name == "SVM":
        c = st.sidebar.slider("c",0.01,10.0)
        params["c"] = c
    else :
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        max_depth = st.sidebar.slider("max_depth",2,10)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
    return params
params = add_parameter(classifier_name)

def add_classifier(params,clf_name):
    if classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors= params["k"])
    elif classifier_name == "SVM":
        clf = SVC(C= params["c"])
    else :
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=1234)
    return clf
clf = add_classifier(params,classifier_name)
# split data into train & test
x_train ,x_test ,y_train , y_test = train_test_split(x,y,test_size=.2,random_state=1234)
# train the data
clf.fit(x_train,y_train)
# get the predict result from model
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)

st.write(f"{classifier_name} accuracy is : {accuracy}")

# plots

pca = PCA()
x_projected = pca.fit_transform(x)

x1 = x_projected[:,0]
x2 = x_projected[:,1]
fig = plt.figure()
plt.scatter(x1,x2,c=y ,alpha=.7 , cmap="viridis" )
plt.xlabel("Principal Component 1 ")
plt.ylabel("Principal Component 2 ")
plt.colorbar()
st.pyplot(fig)



