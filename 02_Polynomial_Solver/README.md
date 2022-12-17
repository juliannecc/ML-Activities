# ML-Activities

## Homework 2
Use SGD to learn polynomial coefficients
## Requirements
To install requirements: 
```
pip install -r requirements.txt
```
## Usage
To get a better view of the code, the process involved as well as the resources used, please see the notebook [act02_clave.ipynb](https://github.com/juliannecc/ML-Activities/blob/main/02_Polynomial_Solver/act02_clave.ipynb). Otherwise,
to use the program with the default settings, download [solver.py](https://github.com/juliannecc/ML-Activities/blob/main/02_Polynomial_Solver/solver.py),
[loader.py](https://github.com/juliannecc/ML-Activities/blob/main/02_Polynomial_Solver/loader.py) and [model.py](https://github.com/juliannecc/ML-Activities/blob/main/02_Polynomial_Solver/model.py) 
then run the command below: 
```
python solver.py
```
To use your own training and testing data, please run the command below: 
```
python solver.py -tr "data_train.csv" -te "data_test.csv"
```
For more information regarding the settings, please run the command below: 
```
python solve.py -h 
```
## Note to reader
Since my machine is not able to run TinyGrad, the code was made using Google Colab, hence, for the best experience, 
use GoogleColab. 

