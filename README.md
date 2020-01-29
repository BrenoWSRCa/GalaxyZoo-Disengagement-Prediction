# Predicting disengagement in a stream of data from a Citizen Science web project

This project was developed in python 3.7.5. To install the requirements, run

```bash
pip install -r requirements.txt
```
There are several notebooks in this project.

In order to get the data set and run the analysis, open the [preprocessing.ipynb]([preprocessing.ipynb]) notebook.

## Running the predictions: 
The prediction notebooks are quite independent of each other.

### Windowed methods:
Run the notebook [windowed_predict.pynb]([windowed_predict.pynb]).

### Recurrent methods:
Run the notebook [rnn_predict.pynb]([rnn_predict.pynb]).