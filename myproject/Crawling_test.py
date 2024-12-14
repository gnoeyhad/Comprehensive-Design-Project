from flask import Flask, render_template, Response, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pathlib import Path 
from ultralytics import YOLO
import pandas as pd 
from pandas.api.types import is_string_dtype
from Crawling import MusinsaScraper
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
        brands = [i.find('span', class_='hds-text-body-medium').text.strip() for i in html.find_all('li', class_='my-16')]
        print(len(brands))

        products = [
            i.text.strip() 
            for div in html.find_all('div', class_='hds-leading-[normal]')
            for i in div.find_all('span', class_='hds-text-gray-primary')
            if 'hds-text-body-medium' in i.get('class', [])
            ]
        print(len(products))

        prices = [
            span.text.strip()
            for div in html.find_all('li', class_='my-16')
            for span in div.find_all('span', class_='hds-text-subtitle-large')
        ]
        print(len(prices))

        # 상품 링크 추출
        links = [
            a['href']
            for div in html.find_all('li', class_='my-16')  # class='sc-kOHUsU'인 div 태그 찾기
            for a in div.find_all('a', class_='items-center')  # 해당 div 안에서 class='gtm-select-item'인 a 태그 찾기
            if 'href' in a.attrs  # a 태그에 href 속성이 있는 경우만
        ]
        print(len(links))

        # 상품 이미지 링크 추출
        img_links = [
            img['srcset']
            for div in html.find_all('div', class_='hds-relative')  # class='eDPFhE'인 div 찾기
            for img in div.find_all('source')  # div 안에서 img 태그 찾기
            if 'srcset' in img.attrs  # img 태그에 src 속성이 있는 경우만
        ]
        print(img_links)
        print(len(img_links))

        # DataFrame 생성
        self.df = pd.DataFrame({
            '브랜드': brands, 
            '상품': products, 
            '가격': prices, 
            '링크': links,
            '이미지 링크': img_links})
        
        print(self.df)


    def filter_by_keyword(self, keyword):
        # 특정 키워드가 포함된 상품 필터링
        if keyword == '입술':
            keyword = '립밤'
        self.df['상품'] = self.df['상품'].astype(str)

        # 개수 3개로 제한
        filtered_df = self.df[self.df['상품'].str.contains(keyword, na=False)][:3]

        return filtered_df


url_2 = str('https://www.hwahae.co.kr/rankings?english_name=category&theme_id=2')


if keyword != None:

    if keyword == '입술':
        keyword =  '립밤'
    
    headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'accept-language': 'en-US,en;q=0.9',
    }
    scraper_2 = MusinsaScraper(url_2, headers)

    # 초기화
    filtered_df_2 = pd.DataFrame()
    
    try:
        # Musinsa 데이터 
        scraper_2.fetch_data()
        filtered_df_2 = scraper_2.filter_by_keyword(keyword)
        print(filtered_df_2)

    except Exception as e:
        print(f"Error with scraper 2: {e}")


    prd = filtered_df_2.to_dict('records')