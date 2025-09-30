#################################################################
###################### MAKEFILE #################################
#################################################################

# Adiciona um marcador para que o make não confunda os alvos com nomes de arquivos
.PHONY: env install install-dev create-jupyter-kernel kedro-init mlflow-ui pipeline pipelines clean clean-data reset

# Versão do Python para o ambiente virtual
PYTHON_VERSION ?= 3.13

# Nome do ambiente
ENV_NAME = .venv

# Arquivos de dependências
REQ_FILE = requirements.txt
REQ_DEV_FILE = requirements-dev.txt

# Caminho do ativador do venv (padronizado para 'bin' para maior compatibilidade)
VENV_ACTIVATE = $(ENV_NAME)/Scripts/activate

# Pipeline rodada
PIPELINE ?= __default__

# --- Regras Principais ---

# Criação do ambiente com uv usando a versão especificada do Python
env:
	@echo "--- Criando ambiente virtual com Python $(PYTHON_VERSION) ---"
	uv venv $(ENV_NAME) -p $(PYTHON_VERSION)

# Ativar env e instalar requirements
install:
	@echo "--- Instalando dependências de produção ---"
	. $(VENV_ACTIVATE) && uv pip install -r $(REQ_FILE) --pre

# Ativar env e instalar requirements de desenvolvimento
install-dev:
	@echo "--- Instalando dependências de desenvolvimento ---"
	. $(VENV_ACTIVATE) && uv pip install -r $(REQ_FILE) -r $(REQ_DEV_FILE)

# Criar kernel do Jupyter para o ambiente
create-jupyter-kernel:
	@echo "--- Criando kernel do Jupyter ---"
	python -m ipykernel install --user --name=$(ENV_NAME) --display-name "Python ($(ENV_NAME))"

# --- Comandos do Projeto ---

# Inicialização do Kedro
kedro:
	. $(VENV_ACTIVATE) && kedro new

# Rodar MLflow UI
mlflow-ui:
	. $(VENV_ACTIVATE) && mlflow ui --backend-store-uri mlruns --port 5000

# Rodar pipeline (por default roda __default__)
run:
	. $(VENV_ACTIVATE) && kedro run --pipeline $(PIPELINE)

# --- Limpeza ---

# Limpeza de ambiente virtual e caches
clean:
	@echo "--- Limpando cache e ambiente virtual ---"
	rm -rf $(ENV_NAME) .pytest_cache .kedro_mlflow .ipynb_checkpoints
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Limpeza de dados intermediários e outputs
clean-data:
	@echo "--- Limpando dados gerados ---"
	rm -rf data/02_intermediate/* data/03_primary/* data/04_feature/* data/05_model_input/* data/06_models/* data/07_model_output/* data/08_reporting/*

# Reset total (cuidado!)
reset: clean clean-data
	@echo "--- Resetando o projeto (removendo também 'mlruns') ---"
	rm -rf mlruns

