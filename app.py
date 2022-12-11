from flask import Flask, jsonify, request, render_template, send_file, redirect, url_for, send_from_directory
from flask_restful import Resource, Api
from flask_cors import CORS
import random
import csv
import os
from os.path import join, dirname, realpath
import pandas as pd
import numpy as np
from tinydb import TinyDB, Query
import json
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from googletrans import Translator
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import emoji
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
import pickle

app = Flask(__name__)
api = Api(app)
CORS(app)

def first_cleaning(column):
    cleaned = re.sub("[^a-zA-Z]+"," ",column).lower() #Remove numeric and punctuation
    additional_word = ["eh","dg", "nya", "tok","kok","kq", "oh" "mah", "gw", "dong","yg", "g", "lg","nya", "dah", "n", "jd", "tsb", "jls", "kt", "kl", "ttg", "klu", "klo", "jwb", "bnyk", "bgtu", "nya", "aja", "sja", "udh", "sudh", "sdh",
                        "ky", "ky", "sperti","x","dgn", "tuk", "d", "utk", "jg", "sma", "sm", "suda","udh", "tp", "kmn", "gmn", "moga", "tpi", "sih", "lah","nih", "tar","gmana", "stlah", "dr",
                        "pa", "yaa", "y", "ya", "yg", "ad", "da", "nya", "sdh", "kalo", "gmn", "knp", "karna", "gimana", "sy", "blum", "bru", "br", "cpt", "kudu", "hrs", "hrus", "bkan","bkn", "dtg","ngk"]
    all_stopwords = stopwords.words("indonesian") + additional_word # Removing stopword and additional unused word
    cleaned= cleaned.split(" ") 
    stemmed_list = [word for word in cleaned if word not in all_stopwords] 
    stemindo = StemmerFactory().create_stemmer().stem(" ".join(stemmed_list)) # Indonesia Stemmer
    trans = Translator(service_urls=["translate.googleapis.com"]).translate(stemindo, dest="en") # Translete to En
    return trans.text

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def second_cleaning(text):
  cleaned = decontracted(text)
  cleaned = re.sub("[^a-zA-Z]+"," ", cleaned).lower() #Remove numeric and punctuation
  cleaned = emoji.demojize(cleaned) # Change Emoji to Word
  cleaned = cleaned.replace("\r", "").replace("\n", " ").replace("\n", " ").lower() # Normalize enter, etc
  return cleaned

class text(Resource):   
    def post(self):
        if not request.json or 'sentences' not in request.json:
            return jsonify({"message" : "Error: No sentences in user input!"})
            
        sentences = request.json["sentences"]
        sentences =  np.array(sentences).tolist()
        
        clean = first_cleaning(sentences)
        clean = second_cleaning(clean)
        with open('modelvec.pkl','rb') as vec:
            vec = pickle.load(vec)

        with open('modelsvm.pkl','rb') as svm:
            svm = pickle.load(svm)
        input_vectorized = vec.transform([clean]) 
        predictions = svm.predict(input_vectorized)

        if list(predictions)[0] == 1.0 or list(predictions)[0] == 2.0:
            out = "Negative"
        else:
            out= "Positive" 
        return jsonify({"result" : out, "num" :  int(list(predictions)[0])})

class csv(Resource):   
    def post(self):
        app.config['UPLOAD_FOLDER'] = 'fileupload'
        csvfile = request.files['rawfile']
        if csvfile.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], csvfile.filename)
            csvfile.save(file_path)
            try:
                data = pd.read_csv("fileupload/" + csvfile.filename, header = None)
                datareal =  pd.read_csv("fileupload/" + csvfile.filename, header = None)
                data.rename(columns = {0:'reviewContent', 1:"rating"}, inplace = True)
                datareal.rename(columns = {0:'reviewContent', 1:"rating"}, inplace = True)

                data["reviewContent"] = data["reviewContent"].apply(first_cleaning)
                data["reviewContent"] = data["reviewContent"].apply(second_cleaning)
                with open('modelvec.pkl','rb') as vec:
                    vec = pickle.load(vec)

                with open('modelsvm.pkl','rb') as svm:
                    svm = pickle.load(svm)

                input_vectorized = vec.transform(data["reviewContent"]) 
                datareal["rating"] = svm.predict(input_vectorized)
                datareal.to_csv("fileupload/" + csvfile.filename)
                return send_file("fileupload/" + csvfile.filename,mimetype='text/csv',download_name="predict "+ csvfile.filename, as_attachment=True)
            except:
                return "There's something wrong with your file!"
            os.remove("fileupload/" + csvfile.filename) 
        else:
            return "Input your file!"

class message(Resource):   
    def post(self):
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"] 

        feedback = {"Name": name,
        "Email": email,
        "Message": message}

        db = TinyDB('feedback.json')
        User = Query()
        db.insert(feedback)

        
        return " ", 204

    
api.add_resource(text,"/text")
api.add_resource(csv, "/csv")
api.add_resource(message, "/message")

if __name__ == "__name__":
    app.run()