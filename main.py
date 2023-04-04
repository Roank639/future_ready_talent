import requests
from bs4 import BeautifulSoup
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
for x in range(1,480):
    soup = get_soup(f"https://www.amazon.in/Prestige-Iris-Grinder-Stainless-Juicer/product-reviews/B0756K5DYZ/ref=cm_cr_getr_d_paging_btm_next_{x}?ie=UTF8&reviewerType=all_reviews&pageNumber={x}")
    print(f'Getting page: {x}')
    get_reviews(soup)
    print(len(reviewlist))
    if not soup.find('li', {'class': 'a-disabled a-last'}):
        pass
    else:
        break
df = pd.DataFrame(reviewlist)
df.to_csv("Amazon_Review_Project_Complete.csv")
print(df)
