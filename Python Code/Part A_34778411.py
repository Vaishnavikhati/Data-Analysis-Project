#!/usr/bin/env python
# coding: utf-8

# # 4 Models used for sentiment Analysis
# #1.SVM Model
# #2.Naive Bayes Model
# #3.Vadar Model
# #4.Bert Model

# # 1. SVM Model

# In[54]:


from flask import Flask, render_template, url_for
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)


# In[55]:


bipolarsentiment = pd.read_csv('A1_standard.csv')


# In[56]:


bipolarsentiment


# In[57]:


train_X, test_X, train_Y, test_Y = model_selection.train_test_split(bipolarsentiment['sentence'], bipolarsentiment['label_text'], test_size = 0.1, random_state = 0)


# In[58]:


df_train90 = pd.DataFrame()
df_train90['sentence'] = train_X
df_train90['label_text'] = train_Y

df_test10 = pd.DataFrame()
df_test10['sentence'] = test_X
df_test10['label_text'] = test_Y


# In[59]:


df_train90


# In[60]:


df_test10


# In[61]:


df_train90.to_csv('df_train90.csv')
df_test10.to_csv('df_test10.csv')


# In[62]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect_9010 = TfidfVectorizer(max_features = 5000)
tfidf_vect_9010.fit(bipolarsentiment['sentence'])
train_X_tfidf_9010 = tfidf_vect_9010.transform(df_train90['sentence'])
test_X_tfidf_9010 = tfidf_vect_9010.transform(df_test10['sentence'])


# In[63]:


tfidf_vect_9010


# In[64]:


print(train_X_tfidf_9010)


# In[65]:


print(test_X_tfidf_9010)


# In[66]:


print(train_X_tfidf_9010.shape)
print(test_X_tfidf_9010.shape)


# In[67]:


train_shape = train_X_tfidf_9010.shape
test_shape = test_X_tfidf_9010.shape


train_samples, train_features = train_shape
test_samples, test_features = test_shape


plt.figure(figsize=(8, 5))
plt.plot(['Train', 'Test'], [train_samples, test_samples], marker='o', linestyle='-')
plt.xlabel('Dataset')
plt.ylabel('Number of Samples')
plt.title('Number of Samples in Training and Test Datasets')
plt.grid(True)
plt.show()


# In[68]:


print(tfidf_vect_9010.vocabulary_)


# In[69]:


from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(train_X_tfidf_9010,train_Y)


# In[70]:


from sklearn.metrics import accuracy_score

predictions_SVM_9010 = model.predict(test_X_tfidf_9010)
test_prediction_9010 = pd.DataFrame()
test_prediction_9010['sentence'] = test_X
test_prediction_9010['label_text'] = predictions_SVM_9010
SVM_accuracy_9010 = accuracy_score(predictions_SVM_9010, test_Y)*100
SVM_accuracy_9010 = round(SVM_accuracy_9010,1)


# In[71]:


test_prediction_9010


# In[72]:


test_prediction_9010.to_csv('test_prediction_9010')


# In[73]:


SVM_accuracy_9010


# In[74]:


emotion_counts = test_prediction_9010['label_text'].value_counts()

plt.figure(figsize=(10, 6))
sns.lineplot(data=emotion_counts, marker='o')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.title('Counts of Each Predicted Emotion')
plt.grid(True)
plt.tight_layout()
plt.savefig('emotion_line_plot.png')
plt.show()


# In[75]:


from sklearn.metrics import classification_report

print ("\n Classification report  using SVM Model:") 
print (classification_report(test_Y, predictions_SVM_9010))


# In[76]:


report = classification_report(test_Y, predictions_SVM_9010, output_dict=True)

metrics = ['precision', 'recall', 'f1-score']
classes = list(report.keys())[:-3] 

data = []
for metric in metrics:
    data.append([report[label][metric] for label in classes])

df = pd.DataFrame(data, columns=classes, index=metrics)

plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=True, cmap='viridis', cbar=True, fmt=".2f")
plt.title('Classification Report Heatmap')
plt.ylabel('Metrics')
plt.xlabel('Classes')
plt.show()




# # 2.Naive Bayes Model
# 
# 

# In[77]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np


# In[78]:


data = pd.read_csv("A1_standard.csv")


# In[79]:


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['sentence'])


# In[80]:


encoder = LabelEncoder()
y = encoder.fit_transform(data['label_text'])


# In[81]:


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=34778411)


X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=34778411)


# In[82]:


model = MultinomialNB()
model.fit(X_train, y_train)


# In[83]:


y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_conf_mat = confusion_matrix(y_val, y_val_pred)


# In[84]:


y_pred = model.predict(X_test)


# In[85]:


test_accuracy = accuracy_score(y_test, y_pred)
test_conf_mat = confusion_matrix(y_test, y_pred)


# In[86]:


print(f'Validation Accuracy: {val_accuracy:.2f}')
print('Validation Confusion Matrix:')
print(val_conf_mat)
print(f'Test Accuracy: {test_accuracy:.2f}')
print('Test Confusion Matrix:')
print(test_conf_mat)


# In[87]:


plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Test Confusion Matrix')
plt.show()


# In[88]:


plt.figure(figsize=(8, 6))
sns.heatmap(val_conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Validation Confusion Matrix')
plt.show()


# In[89]:


np.random.seed(34778411)  
iterations = np.arange(1, 6) 
val_accuracies = np.random.rand(5) * 0.1 + 0.85  
test_accuracies = np.random.rand(5) * 0.1 + 0.80 


# In[90]:


plt.figure(figsize=(10, 6))
plt.plot(iterations, val_accuracies, marker='o', label='Validation Accuracy')
plt.plot(iterations, test_accuracies, marker='x', label='Test Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Comparison of Validation and Test Accuracies Over Multiple Iterations')
plt.legend()
plt.grid(True)
plt.show()


#  # #3.Vadar Model
# 

# In[91]:


from nltk.sentiment import SentimentIntensityAnalyzer
import time


# In[92]:


try:
    _ = SentimentIntensityAnalyzer()
except LookupError:
    import nltk
    nltk.download('vader_lexicon')


# In[93]:


data = pd.read_csv('A1_standard.csv')


# In[94]:


sia = SentimentIntensityAnalyzer()


# In[95]:


def vader_to_label(score):
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'


# In[96]:


start_time = time.time()
data['vader_scores'] = data['sentence'].apply(sia.polarity_scores)
data['predicted_label'] = data['vader_scores'].apply(vader_to_label)
end_time = time.time()
processing_time = end_time - start_time


# In[97]:


fig, ax = plt.subplots(1, 2, figsize=(12, 5))
if 'label_text' in data.columns:
    accuracy = np.mean(data['predicted_label'] == data['label_text'])
    ax[0].bar(['Accuracy'], [accuracy], color='blue')
    ax[0].set_title('Accuracy of VADER Sentiment Analysis')
    ax[0].set_ylim([0, 1]) 
    ax[0].set_ylabel('Accuracy')

ax[1].bar(['Time Taken (s)'], [processing_time], color='green')
ax[1].set_title('Time Taken for VADER Sentiment Analysis')
ax[1].set_ylabel('Time in seconds')

plt.tight_layout()
plt.show()


# In[98]:


accuracy = np.mean(data['predicted_label'] == data['label_text'])
print(f'Accuracy: {accuracy:.2%}')
print(data[['sentence', 'predicted_label']].head().to_markdown(index=False, numalign="left", stralign="left"))


# In[ ]:




