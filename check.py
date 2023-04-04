import streamlit as st
import pandas as pd
import requests
import check
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
# Create a text input widget for the user to enter their input
st.title("Amazon review system ")
user_input = st.text_input('Enter your input link:')

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
}
def get_soup(url):
    r = requests.get(url,headers=headers)
    params={'url': url, 'wait': 2}
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup
reviewlist = []
def get_reviews(soup):
    reviews = soup.find_all('div', {'data-hook': 'review'})
    try:
        for item in reviews:
            review = {
            #'product': soup.title.text.replace('Amazon.ca:Customer reviews: ', '').strip(),
            #'date': item.find('span', {'data-hook': 'review-date'}).text.strip(),
            'title': item.find('a', {'data-hook': 'review-title'}).text.strip(),
            #'rating':  float(item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()),
            'body': item.find('span', {'data-hook': 'review-body'}).text.strip(),
            }
            reviewlist.append(review)
    except:
        pass
for x in range(1,100):
    soup = get_soup(user_input)
    print(f'Getting page: {x}')
    get_reviews(soup)
    print(len(reviewlist))
    if not soup.find('li', {'class': 'a-disabled a-last'}):
        pass
    else:
        break

df = pd.DataFrame(reviewlist)
df.to_csv("Amazon_Review_Project_Complete.csv")
def data(df=df):
    return df
