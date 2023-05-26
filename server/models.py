from sqlalchemy import Column, Integer, String, Float, DateTime
from Connection import Base

class sensor(Base):
    __tablename__ = 'sensor_data'
    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    device_time = Column(Integer, nullable=False)
    acc_x = Column(Float, nullable=True)
    acc_y = Column(Float, nullable=True)
    acc_z = Column(Float, nullable=True)
    gyro_x = Column(Float, nullable=True)
    gyro_y = Column(Float, nullable=True)
    gyro_z = Column(Float, nullable=True)
    mag_x = Column(Float, nullable=True)
    mag_y = Column(Float, nullable=True)
    mag_z = Column(Float, nullable=True)
    roll = Column(Float, nullable=True)
    pitch = Column(Float, nullable=True)
    yaw = Column(Float, nullable=True)
    name = Column(String(16), nullable=False)  # 변경
    age = Column(Integer, nullable=False)
    gender = Column(String(16), nullable=False)  # 변경
    bread = Column(String(16), nullable=False)  # 변경
    kg = Column(Float, nullable=False)

class output(Base):
    __tablename__ = 'prediction_data'
    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    result=Column(String(16), nullable=True)
