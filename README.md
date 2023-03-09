# Water level prediction with various machine learning algorithms for a lowland river

This repository contains code for the paper 
_Water level prediction with various machine learning algorithms for a lowland river_ 
by Zsolt Vizi, Bálint Batki, Luca Rátki, Szabolcs Szalánczi, István Fehérváry, Péter Kozák and Tímea Kiss.


## Environment

The source code for this project is implemented in Python.
The codes for calculations are included in the `src` folder and the plotting is written in Jupyter notebook.

You can download the Jupyter notebook `all_analysis_for_paper.ipynb` for reproducing plots in the paper from
[here](https://drive.google.com/drive/folders/1YHXWRqf8B82foeRUnhrgyxxu3-y8boY-?usp=sharing). 
In this folder, you can find the data used for the visualization and the statistical analysis, 
but the code in the notebook downloads them automatically in the folder `data`.

Requirements:
- Python 3.8+
- packages listed in `requirements.txt`
- `jupyter` package to run notebooks

## Structure of the project

The repository contains the following folders:
- `data`: the validation data and other, exported tables are downloaded to this folder via running the notebook
- `notebooks`: put the notebook named `all_analysis_for_paper.ipynb` here 
(all imports expect this folder as its location)
- `src`: Python module containing the following submodules:
  - `data` for data processing functionalities
  - `evaluation` for statistical analysis
  - `model` for implemented and tested models (LSTM model, Baseline, Linear and MLP)

The files containing trained weights for the models can be found in the above mentioned 
Google Drive folder.

## Testing models

You can download the Jupyter notebook `test_models.ipynb` from the Google Drive folder and 
put into the `notebooks` folder to test the models implemented for this project.
