pyenv install 3.9.0
pyenv local 3.9.0

python -m venv venv

source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python detection_test.py
