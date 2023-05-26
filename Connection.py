from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

HOSTNAME = '210.125.72.190'
PORT = '3308'
USERNAME = 'minseo'
PASSWORD = 'Dmlab6041!'
DBNAME = 'dmlab'
# DATABASE_URL = f'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DBNAME}'
DATABASE_URL = f'mysql+aiomysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DBNAME}'
engine = create_async_engine(DATABASE_URL, pool_recycle=3600, echo = True)  # 데이터베이스 접속 엔진 생성(echo=True는 찍히는 쿼리를 볼 수 있음)
SessionLocal = sessionmaker(engine, autocommit=False,autoflush=False, expire_on_commit=False, class_=AsyncSession) # 비동기 세션 생성
Base = declarative_base()   # 매핑 선언(처리할 DB의 테이블을 설명하고 해당 테이블에 매핑될 클래스 정의하는 작업)