import pandas as pd
import streamlit as st
import seaborn as sns
import plotly as px
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from PIL import Image
import check
#nltk.download('punkt')
#st.set_page_config(page_title="Amazon product recommandation app")
s=st.text_input("Enter the product name")
#check.func(s)
st.title("Review of "+s)
df=pd.read_csv("Amazon_Review_Project_Complete.csv")
print(df.head())
#nltk.download('punkt')
stop_words = set(stopwords.words('english'))


# Perform sentiment analysis on each review
def sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)

    # Get the polarity score
    polarity = blob.sentiment.polarity

    # Classify the sentiment
    if polarity > 0:
        return int(1)
    else:
        return int(0)
df['sentiment'] = df['title'].apply(sentiment)
x = df['title']
y = df['sentiment']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print(df.head())
t =TfidfVectorizer()
# x_train_tf =t.fit_transform(x_train)
# x_test_tf = t.transform(x_test)
# model = LogisticRegression()
# model.fit(x_train_tf, y_train)
# y_pred1 = model.predict(x_test_tf)
# accuracy = accuracy_score(y_test, y_pred1)
# print('Accuracy:', accuracy)
# print(cassification_report(y_test, y_pred1))
l=SVC(kernel='linear', random_state=0)
if len(df.loc[df['sentiment']==1,'sentiment'])>=len(df.loc[df['sentiment']==0,'sentiment']):
    print("The Product is Good and Purchase it ")
else:
    print("We not recommand you to buy this product")
user_menu=st.sidebar.radio(
    'Products Category on Amazon',
    ('Smartphones',"Watches","Perfumes","Earbuds","Fashion",)
)
st.sidebar.title("Offers on Brands")
st.sidebar.checkbox('boAt')
st.sidebar.checkbox("noise")
st.sidebar.checkbox("Apple")
st.sidebar.checkbox("Samsung")
st.title("Amazon Product Recommandation System")

image=Image.open('image.png')
st.image(image,caption="Amazon review",width=20*20)


if user_menu=="Smartphones":
    st.title(user_menu)
    image1 = Image.open('smartphone.jpg')
    image2=Image.open('iphone.jpg')
    st.image([image2,image1], caption=["Applr iphone 14 pro max","Samsung Ultra"], width=20 * 10)
if user_menu=="Watches":
    st.title(user_menu)
    image1=Image.open('applewatch.jpeg')
    image2=Image.open('boatwatch.jpeg')
    image3=Image.open('noise.jpeg')
    st.image([image1, image2,image3], caption=["Apple Watch","boAt watch","noise watch"],width=20*10)

#image1=Image.open("Grinder.jpg")
#st.image(image1,caption="Prestige Iris 750 Watt Mixer")
st.write("The dataset of the product Top-50 & last 50")
st.table(df.head(50))
st.table(df.tail(50))
st.title("Applying Machine learning modals and NLP techniques")
st.text("LogisticRegression "+"Accuracy :"+str(accuracy_score(y_test, y_pred)))
st.text(classification_report(y_test, y_pred))
st.text("Total Positive reviews : "+str(len(df.loc[df['sentiment']==1,'sentiment'])))
st.text("Total Negative  reviews : "+str(len(df.loc[df['sentiment']==0,'sentiment'])))

if len(df.loc[df['sentiment']==1,'sentiment'])>=len(df.loc[df['sentiment']==0,'sentiment']):
    st.title("Good Product with good features PURCHASE IT!✔️👍")
else:
    st.title("Sorry We not recommand you to buy this product")

print(check.data())