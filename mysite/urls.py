from grpc import Status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.conf.urls import url, include
import nltk, re, pprint, string
from nltk import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import nltk
from django.core.files.storage import FileSystemStorage
import cv2 
import argparse 
import os 
import numpy as np
import pytesseract 
# from PIL import Image 
from numpy import asarray
# import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
import datetime
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
import numpy as np
import pandas as pd
import pickle
import joblib



relavant_dict=["lung cancer","cancer","lung"]
privacy_embedding=["father","brother","sister","uncle"]

tokenizer = Tokenizer()
max_length = 30
padding_type='post'
trunc_type='post'
# feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
model_predictor=load_model('models/cancer_predictor.h5')
with open('models/Logistic.mod', 'rb') as f:
    model = pickle.load(f)
loaded_vectorizer = pickle.load(open('models/vector1.pkl', 'rb'))

model_question=load_model('models/question_classifier.h5')
# checkQuestion_Predictor = load_model('models/CheckQuestion_model.pkl')

# with open('models/CheckQuestion_model.pkl' , 'rb') as f:
#     checkQuestion_Predictor = pickle.load(f)
# with open('models/vector.pkl' , 'rb') as f:
#     vectorModel = pickle.load(f)
# checkQuestion_Predictor = joblib.load("models/CheckQuestion_model.pkl")

# with open('models/CheckQuestion_model.pkl' , 'rb') as f:
#     checkQuestion_Predictor = pickle.load(f)

@api_view(['POST'])
def predict(request):
    print(request)
    gender=request.data['gender']
    alcohol=request.data['alcohol']
    genetic=request.data['genetic']
    smoking=request.data['smoking']
    passiveSmoking=request.data['passiveSmoking']
    chest=request.data['chest']
    breath=request.data['breath']
    wheezing=request.data['wheezing']
    finger=request.data['finger']
    cough=request.data['cough']
    asbastos=request.data['asbastos']
    radon=request.data['radon']
    value=[[int(gender),int(alcohol),int(genetic),int(smoking),int(passiveSmoking),int(chest),
    int(breath),int(wheezing),int(finger),int(cough),int(asbastos),int(radon)]]
    print(value)
    sum=int(alcohol)+int(cough)+int(smoking)+int(passiveSmoking)
    status=""
    status,probability =checkStatus(value,sum)
    return Response({"status":status,"probability":probability})

def checkStatus(value,sum):
    print(value)
    Xnew = value
    ynew=model_predictor.predict(Xnew)
    print(ynew[0][0])
    probability=ynew[0][0]*100-sum
    if(ynew[0][0]>0.5):
        return "Cancer situation Detected", round(probability)
    else:
        return "No Cancer situation",0

@api_view(['POST'])
def checkQuestion(request):
    print(request)
    question=request.data['question']
    print(question)
    status=""
    status = relevancyChecker([question])
    return Response({"status":status})

@api_view(['POST'])
def privacyCheck(request):
    print(request)
    status="No privacy detected"
    return Response({"status":status})

# def relevancy(post):
#   tokenizer.fit_on_texts(post)
#   post_sequence = tokenizer.texts_to_sequences(post)
#   padded_post_sequence = pad_sequences(post_sequence, 
#                                        maxlen=max_length, padding=padding_type, 
#                                        truncating=trunc_type)
#   post_prediction = model_question.predict(padded_post_sequence)
#   label = post_prediction.round().item()
#   relavant ="Not relavant"
#   if label == 0:
#     print("%s : Post is NOT relavant" % post)
#     relavant ="Not relavant"
#   elif label == 1:
#     print("%s : Post is relavant" % post)
#     relavant ="Relavant"
    
#   if label ==0 :
#     words = post[0].split()
#     for item in relavant_dict:
#         for word in words:
#             if item==word:
#                 relavant ="Relavant"
#                 break
#   return relavant

# countvec = CountVectorizer(ngram_range=(1,4), 
#                            stop_words='english',  
#                            strip_accents='unicode',
#                            max_features=1000)


# vectorizer = pickle.load(open("vector.pickel", "rb"))
# vectorizer = pickle.load(open("models/vector.pickel", "rb"))


def relevancyChecker(post):
  post_prediction = model.predict(loaded_vectorizer.transform(post))
  label = post_prediction.round().item()
  relavant ="Not relavant"
  if label == 0:
    print("%s : Post is NOT relavant" % post)
    relavant ="Not relavant"
  elif label == 1:
    print("%s : Post is relavant" % post)
    relavant ="Relavant"            
  print(relavant)
  return relavant

# def classify_utterance(utt):
#     # load the vectorizer
#     loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

#     # load the model
#     loaded_model = pickle.load(open('classification.model', 'rb'))

#     # make a prediction
#     print(loaded_model.predict(loaded_vectorizer.transform([utt])))

urlpatterns = [
  url(r'^predict/$', predict),
  url(r'^checkQuestion/$', checkQuestion),
  url(r'^privacyCheck/$', privacyCheck)
]
    
