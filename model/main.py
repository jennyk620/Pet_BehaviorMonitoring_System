from typing import List
from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.params import Depends
from starlette.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import models, schemas, uvicorn, json, os
from databases import Database
from sqlalchemy import insert
from pydantic import ValidationError, BaseModel
from Connection import SessionLocal
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
import pandas as pd
from DL_model import preprocess_data, predict_and_send_result

app = FastAPI()

class SensorDataList(BaseModel):
    data: list

# 전역 변수를 선언하여 데이터를 저장
global_data_list = []

async def get_db():
    async with SessionLocal() as session:
        yield session

@app.post("/api/data", status_code=201)
async def fetch_data(request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()  # 데이터를 JSON으로 변환
    global_data_list.extend(data)  # 전역 리스트에 데이터 추가

    sensor_data = None
    for item in data:
        sensor_data = models.sensor(**item)
        db.add(sensor_data)
    await db.commit()

    if sensor_data is not None:
        await db.refresh(sensor_data)

    return {"message": "success"}


@app.post("/api/data_end")
async def transfer_complete():
    global global_data_list

    if len(global_data_list) == 0:
        return {"message": "No data to process"}

    df = pd.DataFrame(global_data_list)  # 최종 데이터를 DataFrame으로 변환

    # 데이터 프레임을 전처리하여 예측에 사용할 준비
    preprocessed_data = preprocess_data(df)
    await predict_and_send_result(preprocessed_data)

    global_data_list = []

    return {"message": "Data transfer completed"}


if __name__ == "__main__":
    uvicorn.run("main:app", host='210.125.72.190', port=8080, workers=3)
