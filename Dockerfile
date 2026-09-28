FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout 1000 --retries 10 torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --timeout 1000 --retries 10 -r requirements.txt
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]