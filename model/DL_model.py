import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from Connection import SessionLocal
from warnings import simplefilter
from models import output

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 입력 채널 수와 출력 채널 수가 다른 경우, 스킵 연결을 위해 1x1 컨볼루션 연산을 추가합니다
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        # 입력을 스킵 연결로 사용하기 위해 저장
        identity = x
        # 첫 번째 컨볼루션 블록
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 두 번째 컨볼루션 블록
        out = self.conv2(out)
        out = self.bn2(out)

        # 스킵 연결 추가
        identity = self.shortcut(identity)
        out += identity

        # 최종 출력값
        out = self.relu(out)
        return out


class ResNetLSTMClassificationModel(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(ResNetLSTMClassificationModel, self).__init__()

        # 자이로스코프 ResNet 블록
        self.resnet_gyro = nn.Sequential(
            ResidualBlock1D(3, 6, 5, stride=1),
            ResidualBlock1D(6, 12, 5, stride=1),
            ResidualBlock1D(12, 24, 5, stride=1)  # 추가된 부분
        )

        # 가속도계 ResNet 블록
        self.resnet_acc = nn.Sequential(
            ResidualBlock1D(3, 6, 5, stride=1),
            ResidualBlock1D(6, 12, 5, stride=1),
            ResidualBlock1D(12, 24, 5, stride=1)  # 추가된 부분
        )

        # 자이로스코프 LSTM
        self.lstm_gyro = nn.LSTM(24, 64, num_layers=2)  # input_size 변경: 12 -> 24
        self.tanh1 = nn.Tanh()

        # 가속도계 LSTM
        self.lstm_acc = nn.LSTM(24, 64, num_layers=2)  # input_size 변경: 12 -> 24
        self.tanh2 = nn.Tanh()

        self.fc1 = nn.Linear(128, 256)
        self.tanh3 = nn.Tanh()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.tanh4 = nn.Tanh()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.tanh5 = nn.Tanh()
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, output_shape)

    def forward(self, x):
        x_part = x[:, :, :3]
        print(f"x_part shape: {x_part.shape}")
        x_gyro = self.resnet_gyro(x[:, :, :3].transpose(1, 2))
        x_acc = self.resnet_acc(x[:, :, 3:].transpose(1, 2))
        print("Shape of x[:, :, :3].transpose(1, 2):", x[:, :, :3].transpose(1, 2).shape)

        x_gyro = x_gyro.transpose(1, 2)
        x_gyro, _ = self.lstm_gyro(x_gyro)
        x_gyro = self.tanh1(x_gyro[:, -1, :])

        x_acc = x_acc.transpose(1, 2)
        x_acc, _ = self.lstm_acc(x_acc)
        x_acc = self.tanh2(x_acc[:, -1, :])

        x_concat = torch.cat([x_gyro, x_acc], dim=1)
        x = self.tanh3(self.fc1(x_concat))
        x = self.dropout1(x)
        x = self.tanh4(self.fc2(x))
        x = self.dropout2(x)
        x = self.tanh5(self.fc3(x))
        x = self.dropout3(x)
        out = self.fc4(x)
        return out

def preprocess_data(data):
    # 'name', 'gender', 'bread' 열이 존재하는 경우에만 삭제
    columns_to_drop = ['device_time', 'mag_x', 'mag_y', 'mag_z', 'roll', 'pitch', 'yaw', 'name', 'age', 'gender', 'bread', 'kg']
    df = data.drop(columns_to_drop, axis=1, errors='ignore')

    acc_x_list = []
    acc_y_list = []
    acc_z_list = []
    gyro_x_list = []
    gyro_y_list = []
    gyro_z_list = []

    total_data = []

    # 이상치 처리
    def mark_outliers_as_nan(df, column, multiplier=1.5):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan
        return df

    # 스플라인 보간
    def spline_interpolation(df, column):
        x = df[column].values
        y = df.index.values
        mask = np.isnan(x)
        spl = UnivariateSpline(y[~mask], x[~mask], k=3)
        df[column] = spl(y)
        return df

    # 데이터 프레임 처리
    df_processed = df.copy()
    for column in df_processed.columns:
        df_processed = mark_outliers_as_nan(df_processed, column)
        df_processed = spline_interpolation(df_processed, column)

    if len(df_processed.columns) == 6:
        df_processed.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    sequence_length = 150  # 원하는 시퀀스 길이 설정

    for column in df_processed.columns:
        # 컬럼 데이터를 시퀀스 길이로 분할
        sequences = [df_processed[column][i: i + sequence_length].tolist() for i in range(0, len(df_processed) - sequence_length + 1, sequence_length)]

        # 만들어진 시퀀스를 각 리스트에 추가
        if column == 'acc_x':
            acc_x_list.extend(sequences)
        elif column == 'acc_y':
            acc_y_list.extend(sequences)
        elif column == 'acc_z':
            acc_z_list.extend(sequences)
        elif column == 'gyro_x':
            gyro_x_list.extend(sequences)
        elif column == 'gyro_y':
            gyro_y_list.extend(sequences)
        elif column == 'gyro_z':
            gyro_z_list.extend(sequences)

    for column in df_processed.columns:
        # 실제 리스트 변수를 total_data에 추가
        total_data.append(eval('{}_list'.format(column)))

    for i, data in enumerate(total_data):
        print(f'Shape of data {i}: {np.array(data).shape}')

    total_x_train = np.transpose(total_data, (1, 2, 0))  # 축 변경

    # 스케일러 초기화
    scaler = MinMaxScaler()

    # total_x_train의 shape 확인
    n_samples, seq_length, n_features = total_x_train.shape  # 축 순서 변경

    # total_x_train을 2차원으로 변경
    total_x_train_2d = np.reshape(total_x_train, (n_samples * seq_length, n_features))

    # 스케일링 적용
    total_x_train_scaled_2d = scaler.fit_transform(total_x_train_2d)

    # total_x_train을 다시 3차원으로 변경
    total_x_train = np.reshape(total_x_train_scaled_2d, (n_samples, seq_length, n_features))

    return total_x_train



async def load_model(data):
    if data.size == 0:  # data가 비어있는지 확인
        raise ValueError("Data is empty. Check the preprocessing step.")
    total_data = data[0]
    model = ResNetLSTMClassificationModel(total_data.shape[0], 9)
    model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

async def predict(model, total_data):
    total_data_tensor = torch.tensor(total_data).float()
    with torch.no_grad():
        logits = model(total_data_tensor)  # model의 forward 메서드가 비동기로 동작하도록 수정
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predictions = torch.argmax(probs, dim=-1)
    return predictions

# 예측 함수
async def predict_and_send_result(preprocessed_data):
    # 비동기 세션 생성
    async with SessionLocal() as session:
        # 모델 로드
        model = await load_model(preprocessed_data)
        # 예측 수행
        prediction = await predict(model, preprocessed_data)
        print("predict: ", prediction)

        # 각 예측 값을 개별적으로 데이터베이스에 저장
        for p in prediction:
            new_output = output(result=p.item())  # NumPy scalar를 파이썬 scalar로 변환하여 저장
            session.add(new_output)
        await session.commit()
