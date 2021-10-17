import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
#from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
import joblib
from nltk.stem import WordNetLemmatizer


## Item - Item Based system

finaldf = pd.read_csv('data/sample30.csv')
rec_df = finaldf[['id','reviews_rating','reviews_username']]

rec_df.drop_duplicates(keep = 'first', inplace = True)
rec_df = rec_df.groupby(['id','reviews_username']).mean().reset_index()

train,test = train_test_split(rec_df,test_size=.30,random_state=25)

df_pivot = train.pivot(
    index='reviews_username', columns='id', values='reviews_rating'
).T

mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


dummy_train = train.copy()

dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

dummy_train = dummy_train.pivot(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
).fillna(1)

item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0

item_correlation[item_correlation<0]=0

item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)

item_final_rating = np.multiply(item_predicted_ratings,dummy_train)

##  Intergrating Setimental analysis with Recomender system

final_model = joblib.load ('models/Log_reg_model.pkl')

finaldf['reviews_text'] = finaldf.reviews_text.apply(lambda x : x.lower())
#Removing sigle characters/digits
finaldf['reviews_text'] = finaldf.reviews_text.apply(lambda x : re.sub(r'(?:^|\s)(?:\w{1}|[0-9])\s+', ' ', x))
#Removing specail character
finaldf['reviews_text'] = finaldf.reviews_text.apply(lambda x : re.sub(r'[^\w\s]', '', x))
#Removing Digits
finaldf['reviews_text'] = finaldf.reviews_text.apply(lambda x : re.sub(r'[0-9]', '', x))
#Removing multiple white spaces
finaldf['reviews_text'] = finaldf.reviews_text.apply(lambda x : re.sub(r'\s+\s', ' ', x))


stem = WordNetLemmatizer()

for review in finaldf.reviews_text:
    i = 0
    line = []
    for word in review.split():
        #print((stem.lemmatize(re.sub(r'\W', '', word))))
        line.append(stem.lemmatize(word))
    finaldf.reviews_text[i] = ' '.join(line)
    i = +1


X = finaldf.reviews_text
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(X)

def sent_anlysis_on_item(rec_items):
    final_result = {}
    for id,pred_rating in rec_items.items():
        review_txt = finaldf.reviews_text[finaldf.id == id]
        word_features = word_vectorizer.transform(review_txt)
        result = final_model.predict(word_features)
        Pos_per = round(((len(result[result == 1 ])/len(result))*100),2)
        final_result[id] = Pos_per
    return final_result

def item_for_user(user):
    final_list = []
    if (user not in finaldf['reviews_username'].to_list()):
        return('User not found')
    items = item_final_rating.loc[user].sort_values(ascending = False)[0:20]
    resultDict = sent_anlysis_on_item(items)
    sorted_result = sorted(resultDict.items(), key=lambda x: x[1],reverse=True)
    #print(sorted_result[0:5])
    for key,item in sorted_result[0:5] :
        final_list.append(finaldf.name[finaldf.id == key][0:1].values[0])
    return(final_list)

#itsm_list = item_for_user('laura')
#print(itsm_list)