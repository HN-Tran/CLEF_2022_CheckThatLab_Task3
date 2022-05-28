# CLEF 2022 CheckThatLab Task3

This is the repository of team **ur-iw-hnt**.  
Our fine-tuned models are available here: [Hugging Face Repository](https://huggingface.co/hntran/CLEF_2022_CheckThatLab_Task3)  
All TSV files of every model are available here.  
To check the test results, just run this (e.g., for the file `extractive-t5-3b_en.tsv`):  

```
from sklearn.metrics import f1_score
import pandas as pd

y_true = pd.read_csv('English_data_test_release_with_rating.csv')
y_pred = pd.read_csv('extractive-t5-3b_en.tsv', sep='\t')
y_true['our rating'] = y_true['our rating'].replace({"FALSE": 0, "false":0, "False":0, "true":1, "True":1, "TRUE": 1, "Other": 2, "other": 2, "partially false": 3, "Partially False": 3, "Partially false": 3})
y_pred['predicted_rating'] = y_pred['predicted_rating'].replace({"FALSE": 0, "false":0, "False":0, "true":1, "True":1, "TRUE": 1, "Other": 2, "other": 2, "partially false": 3, "Partially False": 3, "Partially false": 3})

f1_score(y_true['our rating'], y_pred['predicted_rating'], average='macro')

## Output should be:
# 0.3953562156909305
```
