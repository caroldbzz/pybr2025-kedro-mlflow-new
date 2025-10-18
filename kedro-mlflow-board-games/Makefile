#################################################################
###################### MAKEFILE #################################
#################################################################

.PHONY: env install install-dev create-jupyter-kernel kedro mlflow-ui run \
        clean clean-data reset conda-remove

# Versão do Python (usaremos no conda create)
PYTHON_VERSION ?= 3.13

# Nome do ambiente conda
ENV_NAME ?= kedro-mlflow-boards-py$(PYTHON_VERSION)
# Versão "cotada" para passar ao conda
QUOTED_ENV := "$(ENV_NAME)"

# Binário do gerenciador (troque por mamba/micromamba se preferir)
CONDA ?= conda

# Arquivos de dependências
REQ_FILE ?= requirements.txt
REQ_DEV_FILE ?= requirements-dev.txt

# Pipeline a ser rodada (Kedro)
PIPELINE ?= __default__

# --- Regras Principais ---

# Cria (se não existir) o env conda com Python e instala uv dentro dele
env:
	@echo --- Criando ambiente conda $(QUOTED_ENV) com Python $(PYTHON_VERSION) ---
	-$(CONDA) create -y -n $(QUOTED_ENV) python=$(PYTHON_VERSION)
	@echo --- Atualizando pip e instalando uv no ambiente ---
	$(CONDA) run -n $(QUOTED_ENV) python -m pip install -U pip
	$(CONDA) run -n $(QUOTED_ENV) python -m pip install -U uv

# Instala deps de produção usando uv (dentro do conda env)
install: env
	@echo --- Instalando dependências de produção com uv ---
	$(CONDA) run -n $(QUOTED_ENV) uv pip install -r $(REQ_FILE) --pre

# Instala deps de produção + desenvolvimento usando uv
install-dev: env
	@echo --- Instalando dependências de desenvolvimento com uv ---
	$(CONDA) run -n $(QUOTED_ENV) uv pip install -r $(REQ_FILE) -r $(REQ_DEV_FILE)

# Cria kernel Jupyter apontando para o ambiente conda
create-jupyter-kernel: env
	@echo --- Criando kernel do Jupyter ---
	$(CONDA) run -n $(QUOTED_ENV) python -m ipykernel install --user \
		--name=$(ENV_NAME) --display-name="Python ($(ENV_NAME))"

# --- Comandos do Projeto ---

kedro: env
	$(CONDA) run -n $(QUOTED_ENV) kedro new

mlflow-ui: env
	$(CONDA) run -n $(QUOTED_ENV) mlflow ui --backend-store-uri mlruns --port 5000

kedro-mlflow:
	$(CONDA) run -n $(QUOTED_ENV) kedro mlflow init

package:
	$(CONDA) run -n $(QUOTED_ENV) kedro package

run: env
	$(CONDA) run -n $(QUOTED_ENV) kedro run --pipeline $(PIPELINE)

# --- Limpeza ---

clean:
	@echo --- Limpando caches e artefatos locais ---
	rm -rf .pytest_cache .kedro_mlflow .ipynb_checkpoints
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-data:
	@echo --- Limpando dados gerados ---
	rm -rf data/02_intermediate/* data/03_primary/* data/04_feature/* \
	       data/05_model_input/* data/06_models/* data/07_model_output/* \
	       data/08_reporting/*

reset: clean clean-data
	@echo --- Resetando o projeto (removendo também 'mlruns') ---
	rm -rf mlruns

conda-remove:
	@echo --- Removendo ambiente conda $(QUOTED_ENV) ---
	$(CONDA) env remove -y -n $(QUOTED_ENV)
