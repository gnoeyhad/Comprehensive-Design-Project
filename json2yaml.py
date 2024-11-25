import json
import os

# 하위 폴더 모두를 건드리기 위해서 반복 실행
for i in range(1, 1101):
    if len(str(i))==1:
        path = '000' + str(i) + '/'
    elif len(str(i))==2:
        path = '00' + str(i) + '/'
    elif len(str(i))==3:
        path = '0' + str(i) + '/'
    else:
        path = str(i) + '/'

    if os.path.isdir(path):
        file_list = os.listdir(path)
        file_list_py = [file for file in file_list if file.endswith('.json')]

        def convert(size, box):  # box: coco형식 xmin , ymin , w , h
            # print('box :', box)
            dw = 1 / size[0]
            dh = 1 / size[1]
            w = box[2]
            h = box[3]
            x = box[0] + w / 2
            y = box[1] + h / 2
            x = round(x * dw, 6)
            w = round(w * dw, 6)
            y = round(y * dh, 6)
            h = round(h * dh, 6)
            if w < 0 or h < 0:
                return False
            return (x, y, w, h)

        data= []

        for i in file_list_py:
            with open (path+i,"r",encoding='UTF8') as f:
                data = json.load(f)

                size = [data['images']['width'],data['images']['height']]

                # bounding box가 있다면
                # 전문가 data만 담긴 경우에는 bbox가 없는 경우도 있음
                # ex: 0002_02_L_03.json의 경우

                b = data['images']['bbox']
                
                if b != None:
                    bb = convert(size, b)
                    file_number = f'{path}/{i[:-5]}.txt'
                    label_file = open(file_number, 'w', encoding='UTF8')  # 파일 생성 및 open
                    label_number = data['images']['facepart']
                    line = f'{label_number} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n'
                    label_file.write(line)
                    label_file.close()
                
                # bbox의 정보가 null인 경우 그냥 안바꾸고 넘어감 
                else:
                    continue
    else:
        continue

print('finish')