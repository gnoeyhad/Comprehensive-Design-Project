import requests
import pandas as pd
from bs4 import BeautifulSoup

class MusinsaScraper:
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers
        self.df = pd.DataFrame()  # 빈 DataFrame 생성

    def fetch_data(self):
        response = requests.get(self.url, headers=self.headers)
        html = BeautifulSoup(response.text, 'html.parser')

        # 데이터 추출
        brands = [i.text.strip() for i in html.find_all('span', class_='sc-dcJtft')]
        products = [i.text.strip() for i in html.find_all('p', class_='sc-gsFSjX')]
        prices = [i.find('span', class_='IOESYE').text.strip() for i in html.find_all('span', class_='sc-dcJtft')]
        
        print(brands, products, prices)
        
        # 상품 링크 추출
        links = [
            i.find('a')['href'] if i.find('a') else None 
            for i in html.find_all('p', class_='gtm-select-item')  # '상품' p 태그에서 a 태그 찾기
        ]

        # DataFrame 생성
        self.df = pd.DataFrame({'브랜드': brands, '상품': products, '가격': prices, '링크': links})

    def filter_by_keyword(self, keyword):
        # 특정 키워드가 포함된 상품 필터링
        return self.df[self.df['상품'].str.contains(keyword, na=False)]

class OliveYoungScraper:
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers
        self.df = pd.DataFrame()  # 빈 DataFrame 생성

    def fetch_data(self):
        response = requests.get(self.url, headers=self.headers)
        html = BeautifulSoup(response.text, 'html.parser')

        # 데이터 추출
        brands = [i.text.strip() for i in html.find_all('span', class_='tx_brand')]
        products = [i.text.strip() for i in html.find_all('p', class_='tx_name')]
        prices = [i.find('span', class_='tx_num').text.strip() for i in html.find_all('span', class_='tx_cur')]

        # 상품 링크 추출
        links = [
            i.find('a')['href'] if i.find('a') else None 
            for i in html.find_all('p', class_='tx_name')  # '상품' p 태그에서 a 태그 찾기
        ]

        # DataFrame 생성
        self.df = pd.DataFrame({'브랜드': brands, '상품': products, '가격': prices, '링크': links})

    def filter_by_keyword(self, keyword):
        # 특정 키워드가 포함된 상품 필터링
        if keyword == '입술':
            keyword = '립밤'
        self.df['상품'] = self.df['상품'].astype(str)
        filtered_df = self.df[self.df['상품'].str.contains(keyword, na=False)]

        return filtered_df
