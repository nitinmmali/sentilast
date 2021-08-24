import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from flask import Flask, render_template, request
df=pd.read_excel(r'Glassdoor Reviews_SmartData.xlsx')
df1=pd.read_excel(r'Modified_Glassdoor_review_copy.xlsx')
def preprocessing_data(x):
    text = request.form['text']
    nltk.download('stopwords')
    all_review=re.sub('[^a-zA-Z]', ' ',text)
    all_review=all_review.lower()
    all_review=all_review.split()
    ps_all=PorterStemmer()
    all_review=[ps_all.stem(word) for word in all_review if word not in set(stopwords.words('english'))]
    all_review= ' '.join(all_review)
    return all_review
Ideal_score=4.5
def Number_of_reviews_required():
    sum_of_all_ratings=np.sum(df1['Overall Satisfaction'])
    Total_Number_of_Ratings=len(df1)
    available_number_of_reviews= 710
    Avg_of_all_Ratings= ((sum_of_all_ratings)/(Total_Number_of_Ratings))
    x=(((Ideal_score*Total_Number_of_Ratings)-(Avg_of_all_Ratings*Total_Number_of_Ratings))/(0.5))
    return round(x)
drop=0.5
# Ratings assumed are 3* ratings
Ratings_assumed=3
sum_of_all_ratings=np.sum(df1['Overall Satisfaction'])
Total_Number_of_Ratings=len(df1)
Avg_of_all_Ratings= ((sum_of_all_ratings)/(Total_Number_of_Ratings))
def Number_of_reviews_for_Drop():
    x=3.7-drop
    reviews_for_Drop= ((Avg_of_all_Ratings-x)*710) / ((Avg_of_all_Ratings - drop) - (Ratings_assumed))
    return round(reviews_for_Drop)