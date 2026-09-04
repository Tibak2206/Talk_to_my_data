FROM python:3.14-slim

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ src/
COPY app/ app/
COPY data/raw/credit_card_default.csv data/raw/credit_card_default.csv
COPY models/gradient_boosting_model.joblib models/gradient_boosting_model.joblib

EXPOSE 8080

CMD streamlit run app/streamlit_app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableXsrfProtection=false
