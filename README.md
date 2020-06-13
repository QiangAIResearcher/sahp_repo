# SAHP

This is the repository for the self-attentive Hawkes process where self-attention is used to adapt the intensity function of Hawkes process.

## Dataset
The experiment datasets are available on this [Google drive] (https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U). To run the model, you should download them to
the parent directory of the source code, with the folder name `data`

## Package
The Python version should be at least 3.5 and the torch version can be 0.4.1

## Scripts
`models` defines the self-attentive Hawkes model, multi-head attention and the related.

`main_func.py` is the main function to run the experiments, hyper-parameters are provided here.

`utils` contains utility functions

To run the model: python main_func.py
