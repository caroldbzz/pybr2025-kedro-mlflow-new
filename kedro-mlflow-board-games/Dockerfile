# Usa a imagem base Python 3.13 slim
FROM python:3.13-slim

# Define o diretório de trabalho
WORKDIR /home/kedro

# Copia o arquivo de requisitos e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código fonte do projeto
COPY . .

# Define o ambiente padrão do Kedro
ENV KEDRO_ENV=local

# Define a variável de ambiente para o URI de rastreamento do MLflow dentro do container
ENV MLFLOW_TRACKING_URI=file:/home/kedro/data/mlruns

# Comando padrão para execução
CMD ["kedro", "run"]