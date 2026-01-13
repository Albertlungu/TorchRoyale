pyenv install 3.12
pyenv local 3.12

python -m venv venv

source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
