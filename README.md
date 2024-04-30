## Evaluating the Impact of Local Differential Privacy on Utility Loss via Influence Functions

This repository presents all the code used in the experimentation section of _Evaluating the Impact of Local Differential Privacy on Utility Loss via Influence Functions_ published in IJCNN 2024. This repository is broken down into three sections:

1. data
2. results
   - adult
   - acspubcov
   - mnist
3. all other files

The data folder houses all the data used in the experimentation. The results folder contains three separate folders (adult, acspubcov, mnist) each of which correspond to the output folder of the specific dataset. For example, _results/adult_ will contain the pickled results for adult-labels-flc.ipynb and adult-labels.ipynb. All of the jupyter notebook files are self contained, and should be able to run without modification. The name of the file details which dataset it uses, and whether randomization is applied to the features, the labels, or the features and labels, or if forward loss correction is applied. 

The required package versions used in the experiments are:

- tqdm: 4.64.1
- numpy: 1.23.5
- scipy: 1.9.3
- pandas: 1.5.2
- python: 3.9.15
- pytorch: 1.13.0
- matplotlib: 3.6.2
- folktables: 0.0.12
- scikit-learn: 1.1.3
- scienceplots: 2.0.1
