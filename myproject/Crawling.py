import requests
import pandas as pd
from bs4 import BeautifulSoup

class HwahaeScraper:
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers
        self.df = pd.DataFrame()  # 빈 DataFrame 생성

    def fetch_data(self):
        response = requests.get(self.url, headers=self.headers)
        html = BeautifulSoup(response.text, 'html.parser')

        # 데이터 추출
        brands = [i.find('span', class_='hds-text-body-medium').text.strip() for i in html.find_all('li', class_='my-16')]

        products = [
            i.text.strip() 
            for div in html.find_all('div', class_='hds-leading-[normal]')
            for i in div.find_all('span', class_='hds-text-gray-primary')
            if 'hds-text-body-medium' in i.get('class', [])
            ]

        prices = [
            span.text.strip()
            for div in html.find_all('li', class_='my-16')
            for span in div.find_all('span', class_='hds-text-subtitle-large')
        ]

        # 상품 링크 추출
        links = [
            a['href']
            for div in html.find_all('li', class_='my-16')  # class='sc-kOHUsU'인 div 태그 찾기
            for a in div.find_all('a', class_='items-center')  # 해당 div 안에서 class='gtm-select-item'인 a 태그 찾기
            if 'href' in a.attrs  # a 태그에 href 속성이 있는 경우만
        ]

        # 링크가 상대주소로 되어있기에 절대주소로 변경 
        for i in range(len(links)):
            links[i] = 'https://www.hwahae.co.kr/' + links[i]

        # 상품 이미지 링크 추출
        img_links = [
            img['srcset']
            for div in html.find_all('div', class_='hds-relative')  # class='eDPFhE'인 div 찾기
            for img in div.find_all('source')  # div 안에서 img 태그 찾기
            if 'srcset' in img.attrs  # img 태그에 src 속성이 있는 경우만
        ]

        # 개수를 3개로 제한
        brands = brands[:3]
        products = products[:3]
        prices = prices[:3]
        links = links[:3]
        img_links = img_links[:3]

        # DataFrame 생성
        self.df = pd.DataFrame({
            '브랜드': brands, 
            '상품': products, 
            '가격': prices, 
            '링크': links,
            '이미지 링크': img_links})
        

    def filter_by_keyword(self, keyword):
        # 특정 키워드가 포함된 상품 필터링
        if keyword == '입술 건조':
            keyword = '립밤'
        self.df['상품'] = self.df['상품'].astype(str)

        filtered_df = self.df

        return filtered_df


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
        links = [i.attrs['href'] for i in html.find_all('a' ,class_='prd_thumb')]

        # 이미지 추출
        img_links = [i.find('img').attrs['src'] for i in html.find_all('a', class_='prd_thumb')]

        # DataFrame 생성
        self.df = pd.DataFrame({
            '브랜드': brands, 
            '상품': products, 
            '가격': prices, 
            '링크': links,
            '이미지 링크': img_links})

    def filter_by_keyword(self, keyword):
        # 특정 키워드가 포함된 상품 필터링
        if keyword == '입술':
            keyword = '립밤'
        self.df['상품'] = self.df['상품'].astype(str)

        # 개수 3개로 제한
        filtered_df = self.df[self.df['상품'].str.contains(keyword, na=False)][:3]

        return filtered_df
