FROM python:3.10-slim

ENV PYTHONUNBUFFERED True

COPY face_attendance/service/requirements.txt ./

RUN pip install -r requirements.txt

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY face_attendance $APP_HOME/face_attendance

CMD ["uvicorn", "face_attendance.service.main:app", "--host", "0.0.0.0", "--port", "8080"]
