# Pet_BehaviorMonitoring_system
Capstone Design Project (2023)

## model
#### DMLAB_xy_data.ipynb
선행 연구에서 수집된 반려동물 행동에 대한 센서 데이터(자이로, 가속도, 지자계) 전처리

#### kaggle_xy_data.ipynb
Kaggle에서 제공되는 반려동물 행동에 대한 센서 데이터(자이로, 가속도) 전처리

#### csv_processing.ipynb
모델 학습에 사용하기 위한 csv 생성

#### Capstone_final.ipynb
모델 설계 및 학습 진행


## server
#### Connection.py
DB 연결 관리

#### DL_model.py
데이터 추론에 사용할 학습 모델

#### main.py
센서 데이터 처리하는 API 구현

#### models.py
DB 모델 정의

#### shemas.py
DB 저장을 위한 데이터 모델 정의
