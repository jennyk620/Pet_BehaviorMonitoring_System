from pydantic import BaseModel  #parsing 라이브러리=> 출력 모델의 유형과 제약 조건 보장


class sensor(BaseModel):    # Type Hints:파라미터 값에 어떤 자료형이 들어와야 하는지 명시
    id: int
    device_time: int
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    mag_x: float
    mag_y: float
    mag_z: float
    roll: float
    pitch: float
    yaw: float
    name: str
    age: int
    gender: str
    bread: str
    kg: float

    class Config:
        orm_mode = True

class output(BaseModel):    # Type Hints:파라미터 값에 어떤 자료형이 들어와야 하는지 명시
    id: int
    result: str

    class Config:
        orm_mode = True
