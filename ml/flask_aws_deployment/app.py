#!/usr/bin/env python
# coding: utf-8

# To run the application via rest end point interface

# In[1]:


import flask


# In[2]:


#!pip install flask


# In[2]:


from flask import request
app=flask.Flask(__name__)


# In[6]:


#!pip install flask_cors


# In[3]:


from flask_cors import CORS
CORS(app)


# In[4]:


@app.route('/')
def default():
    return '<h1> API Server is working </h1>'


# In[5]:


@app.route('/predict')
def predict():
    from sklearn.externals import joblib
    model=joblib.load('marriage_age_predic_model.ml')
    #age_predict=model.predict([[1,2,5,6,5,175]])
    age_predict=model.predict([[request.args['gender'],
                                request.args['religion'],
                                request.args['caste'],
                                request.args['mother_tongue'],
                                request.args['country'],
                                request.args['height_cms']
                                ]])
    return str(age_predict)


# In[6]:


app.run(debug=True)


# ##### download this file as app.py
# ##### open command and navigate to the file location app.py
# ##### type
# ##### python app.py
# ##### once the app has started
# ##### open the browser and type  
# ##### http://127.0.0.1:5000
# ##### http://127.0.0.1:5000/predict

# In[ ]:




