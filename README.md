# T-DEV-810

AI: Image recognition

## Jupyter notebook
jupyter only works with python2.7

`cd ./T-DEV-810/`

`pip install virtualenv`

`virtualenv -p python2.7 venv2.7`

`source venv2.7/bin/activate` : to activate virtual environment

`pip install -r requirements2.7.txt`

`jupyter notebook notebook-1.ipynb`

`deactivate` : deactivate virtual environment

## Using Virtualenv
### Install virtual env
`pip install virtualenv`

`cd ./T-DEV-810/`

### Init virtualenv
`virtualenv -p python3.7 venv3.7`

### Use existing virtualenv
`source venv3.7/bin/activate`

### Install package in virtualenv
`pip install <module>`

### Save dependencies
`pip freeze > requirements3.7.txt`

### Load dependencies
`pip install -r requirements3.7.txt`

### Stop using virtualenv
`deactivate`

## Tensorboard
### Activate logs
Edit init_or_load_then_train_main.py :
    CnnModel(filename, inputShape, 3, False) => CnnModel(filename, inputShape, 3, True)

### Train with activate logs
`python init_or_load_then_train_main.py`

### Launch tensorboard server
`tensorboard --logdir logs/fit` : default:: /localhost:6006/
