FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
# lance les tests à l'exécution du conteneur
CMD ["pytest", "-q"]
