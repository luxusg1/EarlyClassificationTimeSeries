# EarlyClassificationTimeSeries
This repository contains an implementation of the paper "A deep reinforcement learning approach for early classification of time series" published in 2018 26th European Signal Processing Conference (EUSIPCO)

## Data
The data used in this project is the GunPoints datasets which comes from the UCR/UEA archive 

## Code
The code is divided as follows:
* The [Train.py](#Train.py) python file contains the necessary code to run the train
* The [Agent.py](#Agent.py) python file contains the necessary code about the Agent (model, memory, ...)
* The [Env.py](#Env.py) python file contains all about the environement (reward, state, step, ...)
* The [utils.py](#utils.py) python file contains all necessary utils function (loss_function and zeros_padding function)
* The [Import.py](#Import.py) python file contains all necessary package to load

To run the training of model you should issue this followinf command:
```
$ python Train.py 

```

## Prerequistes 
All python packages needes are listed in [pip-requirement.txt](#pip-requirement.txt) file and can be installed simply using the pip command:

* [numpy](www.numpy.org)
* [pandas](https://pandas.pydata.org)
* [sklearn](scikit-learn.org/stable/)
* [scipy](https://www.scipy.org)
* [matplotlib](https://matplotlib.org)
* [pyts](https://pypi.org/project/pyts/)
* [tensorflow-gpu](https://www.tensorflow.org)
* [keras](https://keras.io)







