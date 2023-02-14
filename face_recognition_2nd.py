import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


# faces폴더에 있는 파일 리스트 얻기
data_path = 'faces/'

# listdir(입력 경로 내의 모든 파일과 폴더명을 리스트 반환), isfile(파일이 있으면 true, 파일이 아니거나 없으면 false), join(경로, 파일명): 파일명과 경로 합치기
# in listdir(data_path) faces 폴더안의 파일들을 리스트로 만들어준 다음에 만약 join(data_path,f)가 파일 이라면(isfile) f로 받아서 onlyfiles 리스트에 넣어줌
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
# 데이터와 매칭될 라벨 변수
Training_Data, Labels = [], []

# 파일 개수 만큼 루프
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    # 이미지 불러오기
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Training_Data 리스트에 이미지를 바이트 배열로 추가
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    # Labels 리스트엔 카운트 번호 추가
    Labels.append(i)

# Labels를 32비트 정수로 변환
Labels = np.asarray(Labels, dtype=np.int32)
# LBP 얼굴인식기 생성 및 학습
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")

