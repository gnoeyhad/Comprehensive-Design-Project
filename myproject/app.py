from flask import Flask, render_template, Response, request, jsonify, send_file
import os
import cv2
import torch
import time
import re
from werkzeug.utils import secure_filename
from pathlib import Path 
from ultralytics import YOLO
import pandas as pd 
from pandas.api.types import is_string_dtype
from Crawling import OliveYoungScraper, MusinsaScraper
from SkinPredictor import MoisturePorePredictor

app = Flask(__name__, static_folder='static')

model = YOLO('./best.pt')

@app.route('/')
def video_show():
    return render_template('video_show.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/static_frame')
def static_frame():
    return send_file('static/saved_frame.jpg', mimetype='image/jpeg')


# 영상 스트리밍 함수
def gen_frames():
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 5초 후 프레임 저장 및 스트리밍 종료
        elapsed_time = time.time() - start_time # 경과 시간 
        if elapsed_time >= 5: # 경과 시간이 5초를 지났을 때 
            cv2.imwrite('static/saved_frame.jpg', frame) 
            break

        # 스트리밍을 위한 프레임 전송
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n' + 
               f'Elapsed-Time: {elapsed_time}\r\n'.encode())

    
    cap.release()

# 추천 시스템 함수: 회귀를 통한 예측값과 image detection 결과를 비교하여 추천할 화장품을 선정
def recommendation(keyword):
    ## predictions의 기준을 통해서 keyword를 뽑아내는 작업 필요
    url_1 = str('https://www.oliveyoung.co.kr/store/search/getSearchMain.do?query='+keyword+'&giftYn=N&t_page=홈&t_click=검색창&t_search_name='+keyword)
    url_2 = str('https://www.musinsa.com/category/104?keyword='+keyword+'&keywordType=keyword&gf=A' )
    
    if keyword != None:

        if keyword == '입술':
            keyword =  '립밤'
        
        headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'accept-language': 'en-US,en;q=0.9',
    }

        scraper_1 = OliveYoungScraper(url_1, headers)
        scraper_2 = MusinsaScraper(url_2, headers)

        # 초기화
        filtered_df = pd.DataFrame()

        try:
            # Oliveyoung 데이터
            scraper_1.fetch_data()
            filtered_df = scraper_1.filter_by_keyword(keyword)
        except Exception as e:
            print(f"Error with scraper 1: {e}")

        try:
            # Musinsa 데이터 
            scraper_2.fetch_data()
            filtered_df_2 = scraper_2.filter_by_keyword(keyword)
            filtered_df = pd.concat([filtered_df, filtered_df_2], ignore_index=True)
        except Exception as e:
            print(f"Error with scraper 2: {e}")

        # 필터링 된 결과 
        print(scraper_1.filter_by_keyword(keyword))

         # 데이터가 있는 경우만 클리닝 처리
        if not filtered_df.empty:
            for column in filtered_df.columns:
                # 문자열 열만 클리닝
                if is_string_dtype(filtered_df[column]):
                    filtered_df[column] = filtered_df[column].map(
                        lambda x: x.strip() if isinstance(x, str) else x
                    )
        else:
            # 데이터가 없는 경우 None 반환
            return None
        
    return filtered_df


# Regression and prediction route
@app.route('/predict', methods=['POST'])
def skin_prediction():
    try:
        # age와 gender를 사용한 예측 수행
        age = request.form.get('age')
        gender = request.form.get('gender')

        if age is None or gender is None:
            return jsonify({"error": "Invalid input"}), 400

        try:
            age = int(age)
        except ValueError:
            return jsonify({"error": "Invalid age input"}), 400

        data_dir = "C:\\Users\\da010\\Downloads\\028.한국인 피부상태 측정 데이터\\3.개방데이터\\1.데이터\\Validation\\02.라벨링데이터\\VL\\All_json_file\\"
        predictor = MoisturePorePredictor(data_dir)
        
        try:
            predictor.load_models()
        except Exception as e:
            print(f"Error loading models: {e}")
            return jsonify({"error": "Model loading failed"}), 500

        # 예측 수행
        try:
            predictions = predictor.predict(age, gender)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({"error": "Prediction failed"}), 500

        predictions_m = {k: round(float(v), 2) for k, v in predictions.items()}
        print("Processed predictions (predictions_m):", predictions_m)

        # 업로드된 파일 처리
        file = request.files.get('file')
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join('static', 'uploads', filename)
            file.save(image_path)
        else:
            image_path = 'static/saved_frame.jpg'

        # YOLO 모델 바운딩 박스 적용
        try:
            image = cv2.imread(image_path)
            results = model.predict(image)
            annotated_image = results[0].plot()
        except Exception as e:
            print(f"Error in YOLO model prediction: {e}")
            return jsonify({"error": "YOLO model failed"}), 500

        # YOLO 결과 저장 경로
        processed_image_path = os.path.join('static', 'uploads', 'processed_saved_frame.jpg')
        processed_image_path = processed_image_path.replace("\\", "/")  # 경로 수정
        try:
            cv2.imwrite(processed_image_path, annotated_image)
        except Exception as e:
            print(f"Error saving YOLO annotated image: {e}")
            return jsonify({"error": "Failed to save YOLO image"}), 500

        # YOLO 결과에서 클래스 이름, 정확도 추출
        detection_data = []
        try:
            for detection in results[0].boxes:  # YOLOv8에서 바운딩 박스 정보 접근
                confidence = detection.conf[0].item()
                class_id = int(detection.cls[0].item())
                class_name = model.names[class_id]

                # confidence가 0.5 이상일 때만 추가
                if confidence > 0.5:
                    detection_data.append({
                        "class_name": class_name,
                        "confidence": float(confidence)}
                    )
        except Exception as e:
            print(f"Error processing YOLO detections: {e}")
            return jsonify({"error": "Failed to process YOLO detections"}), 500

        # 필터링 키워드 결정 및 추천 항목 가져오기
        ## 수정 중인 부분 
        #itemNum = len(detection_data)
        if detection_data:
            for item in detection_data:
                if item:
                    class_name = item["class_name"]  # Access the 'class_name' key from the dictionary
                    if class_name == 'lip':  # Check if the class_name is 'lip'
                        keyword = '입술'
                    elif class_name == 'fore' or class_name == 'eight' or class_name == 'eye_u' or class_name == 'eye_t' or class_name == 'be':
                        keyword = '주름'
                    elif class_name == 'job':
                        keyword = '잡티'
                    else:
                        keyword = '모공'
            # elif itemNum >= 2:
            #     keyword = []
            #     for item in detection_data:
            #         if item:
            #             class_name = item[class_name] 
            #             if class_name == 'eight' or class_name == 'eye_u' or class_name == 'eye_t':
            #                 keyword.append('주름') 
            #             elif class_name == 'lip':
            #                 keyword.append('입술')
            #             else:
            #                 keyword.append('모공')
            #         else:
            #             keyword.append('모공')
        else:
            keyword = 'None'   

        print(keyword)

        # 필터링된 제품 추천
        try:
            filtered_products = recommendation(keyword)
        except Exception as e:
            print(f"Error in product recommendation: {e}")
            return jsonify({"error": "Recommendation failed"}), 500

        # 추천 결과 처리
        if filtered_products is None:
            return render_template(
                'result.html',
                value=predictions_m,
                processed_image_path=processed_image_path,
                tables=None,
                message="",
                keyword=keyword
            )
        else:
            return render_template(
                'result.html',
                value=predictions_m,
                processed_image_path=processed_image_path,
                tables=[filtered_products.to_html(classes='data')] if not isinstance(filtered_products, str) else None,
                message=filtered_products if isinstance(filtered_products, str) else None,
                keyword=keyword
            )
    except Exception as e:
        print(f"Unhandled error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == "__main__":
    app.run(debug=True)
