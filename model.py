import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
nltk.download("stopwords") 
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

# Memanggil Data
data = pd.read_csv("ReviewLazada.csv")
rate = data[["rating", "reviewContent"]]
rate = rate.dropna()

# Mengambil 2184 data teratas pada masing masing rating
rate_5 = rate[rate["rating"] == 5]
rate_4 = rate[rate["rating"] == 4]
rate_3 = rate[rate["rating"] == 3]
rate_2 = rate[rate["rating"] == 2]
rate_1 = rate[rate["rating"] == 1]

rate_5 = rate[rate["rating"] == 5].drop(rate_5.index[2184:])
rate_4 = rate[rate["rating"] == 4].drop(rate_4.index[2184:])
rate_3 = rate[rate["rating"] == 3].drop(rate_3.index[2184:])
rate_2 = rate[rate["rating"] == 2]
rate_1 = rate[rate["rating"] == 1].drop(rate_1.index[2184:])

# Text Preprocessing Pertama
def first_cleaning(column):
    cleaned = re.sub("[^a-zA-Z]+"," ",column).lower() #Remove numeric and punctuation
    additional_word = ["eh","dg", "nya", "tok","kok","kq", "oh" "mah", "gw", "dong","yg", "g", "lg","nya", "dah", "n", "jd", "tsb", "jls", "kt", "kl", "ttg", "klu", "klo", "jwb", "bnyk", "bgtu", "nya", "aja", "sja", "udh", "sudh", "sdh",
                        "ky", "ky", "sperti","x","dgn", "tuk", "d", "utk", "jg", "sma", "sm", "suda","udh", "tp", "kmn", "gmn", "moga", "tpi", "sih", "lah","nih", "tar","gmana", "stlah", "dr",
                        "pa", "yaa", "y", "ya", "yg", "ad", "da", "nya", "sdh", "kalo", "gmn", "knp", "karna", "gimana", "sy", "ga", "tdk", "gak", "blm","gk", "gx", "tdak", 
                        "nggak", "bs", "bsa", "bgt", "bgtt", "bget", "msh", "msih", "cm", "cuman", "blum", "bru", "br", "cpt", "kudu", "hrs", "hrus", "bkan","bkn", "dtg","ngk"]
    all_stopwords = stopwords.words("indonesian") + additional_word # Remove stopword and additional unused word
    cleaned= cleaned.split(" ") 
    stemmed_list = [word for word in cleaned if word not in all_stopwords] 

    stemindo = StemmerFactory().create_stemmer().stem(" ".join(stemmed_list)) # Indonesia Stemmer
    trans = Translator(service_urls=["translate.googleapis.com"]).translate(stemindo, dest="en") # Translete to English
    return trans.text

rate_1["reviewContent"] = rate_1["reviewContent"].apply(first_cleaning)
rate.to_csv("rateclean1.csv")

rate_2["reviewContent"] = rate_2["reviewContent"].apply(first_cleaning)
rate.to_csv("rateclean2.csv")

rate_3["reviewContent"] = rate_3["reviewContent"].apply(first_cleaning)
rate.to_csv("rateclean3.csv")

rate_4["reviewContent"] = rate_4["reviewContent"].apply(first_cleaning)
rate.to_csv("rateclean4.csv")

rate_5["reviewContent"] = rate_5["reviewContent"].apply(first_cleaning)
rate.to_csv("rateclean5.csv")

# Memanggil Data 
rate = pd.concat([pd.read_csv("rateclean1.csv"), pd.read_csv("rateclean2.csv"), pd.read_csv("rateclean3.csv"), pd.read_csv("rateclean4.csv"), pd.read_csv("rateclean5.csv")], ignore_index=True)  

np.random.seed(100) 
rate = shuffle(rate)
rate = rate.dropna()

# Text Preprocessing Kedua
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
  cleaned = decontracted(text) # Lemmatize
  cleaned = re.sub("[^a-zA-Z]+"," ", cleaned).lower() #Remove numeric and punctuation
  cleaned = emoji.demojize(cleaned) # Change Emoji to Word
  cleaned = cleaned.replace("\r", "").replace("\n", " ").replace("\n", " ").lower() # Normalize enter, etc
  return cleaned

rate["reviewContent"] = rate["reviewContent"].apply(second_cleaning)

# Membangun dan Menguji Model
x = rate.reviewContent
y = rate.rating

modelvec = TfidfVectorizer()

x = modelvec.fit_transform(x.values.astype("U"))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


modelsvm = svm.SVC()
modelsvm.fit(x_train, y_train)

pred = modelsvm.predict(x_test)

# Menyimpan Model
pickle_vec = open("modelvec.pkl", "wb")
pickle.dump(modelvec, pickle_vec)
pickle_vec.close()

pickle_svm = open("modelsvm.pkl", "wb")
pickle.dump(modelsvm, pickle_svm)
pickle_svm.close()