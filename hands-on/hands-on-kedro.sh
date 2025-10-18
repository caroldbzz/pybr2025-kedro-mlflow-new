# Parte 1: Install Kedro
conda create -n <env_name> python=3.13 -y
conda activate <env_name>
pip install uv
uv pip install kedro
kedro info

# Parte 2: Create a new Kedro project
kedro new \
  --name "kedro-exemplo" \
  --tools "lint,test,log,docs,data" \
  --example y