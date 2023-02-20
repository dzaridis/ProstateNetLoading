
# Prostate Net Loader

Prostate Net Loader contains functions to assist the Data Mining and loading process of Patients originated from the ProCAncer-AI Horizon's 2020 project.  
The package is under construction for the time being therefore any suggestion would be appreciated.

## Installation

Install the project via pip or pull the repo

```bash
pip install PROSTATENETLOADER == 0.0.1
```
    
## Usage/Examples Series Parser tool
Detailed explanation of the series parser tool is presented at the ParquetParser_Examples.ipynb
## Usage/Examples PROSTATENETLOADER Module

Examples could be found in Module_Examples.ipynb regarding the package. An example for a single patient is presented below

### Single Patient
a) Import Libraries
```python
import pandas as pd
import SimpleITK as sitk
import ProstateNetLoaders
```
b) Set the patient folder path and the csv extracted by the sequence selector tool
```python
pth = "PCa-..."
metadata= pd.read_csv("results.csv", 
                names=["patient_id", "study_uid", 
                "series_uid", "series_type", "series_type_heuristics"])
```
c) Execute loaders and pick orientation ("AX","COR", "SAG") and sequence ("T2","ADC","DWI") and whether to be AI sequence parser (Heuristics = False) or Heuristics = True
```python
a = ProstateNetLoaders.ExecuteLoader.Execute(pth, metadata,  Heuristics = True) 
a.LoadArrays(orientation="AX", seq="T2")
```
d) Get dictionaries where keys are the series names, values are the Image numpy arrays
```python
pat,ann = a.GetItems() 
```

### Batch Loading 

The structure of the folders should be like this
![image](https://github.com/dzaridis/ProstateNetLoading](https://github.com/dzaridis/ProstateNetLoading/blob/main/Images/dIR.png)

```python
pth_batch = "Patients"
patients = {}
Sequence = "T2" # pick you sequence between "T2", "ADC", "DWI"
T2_absence = [] # Store the names of the failed patients
for patient in os.listdir(pth_batch):
    pat = os.path.join(pth_batch,patient)
    a = ProstateNetLoaders.ExecuteLoader.Execute(pat, metadata)
    try:
        a.LoadArrays(orientation="AX", seq=Sequence)
        pat,ann = a.GetItems()
        patients.update({patient:{Sequence:np.array(list(pat.values())[0]),"Lesion": np.array(list(ann.values())[0])}})
    except: 
        T2_absence.append(patient)
        continue
```
## Authors

- [Dimitris Zaridis](dimzaridis@gmail.com)
- [Harry Kalantzopoulos](xkalantzopoulos@gmail.com)
- [Eugenia Mylona](mylona.eugenia@gmail.com)
- [Nikolaos Tachos](ntachos@gmail.com)
- José Guilherme Almeida


## Badges


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
![Python](https://img.shields.io/badge/Python-3.6-green)


## License

[MIT](https://choosealicense.com/licenses/mit/)


![Logo](https://www.procancer-i.eu/wp-content/uploads/2020/07/logo.png)

