# CLEF 2022 CheckThatLab Task3

This is the repository of team **ur-iw-hnt**. You can read our paper [here](https://ceur-ws.org/Vol-3180/paper-60.pdf).
Our fine-tuned models are available here: [Hugging Face Repository](https://huggingface.co/hntran/CLEF_2022_CheckThatLab_Task3)  
All TSV files of every model are available in the `TSV_outputs` folder.  
To check the test results, just run this (e.g., for the file `extractive-t5-3b_en.tsv`):  

```python
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

Please cite our paper as:
```
@InProceedings{clef-checkthat:2022:task3:Tran_ur-iw-hnt,
  author    = {Tran, Hoai Nam and Kruschwitz, Udo},
  title     = {{ur-iw-hnt at CheckThat! 2022:} Cross-lingual Text Summarization for Fake News Detection},
  year      = {2022},
  booktitle = {Working Notes of {CLEF} 2022 - Conference and Labs of the Evaluation Forum},
  editor = {Faggioli, Guglielmo andd Ferro, Nicola and Hanbury, Allan and Potthast, Martin},
  series    = {CLEF~'2022},
  address   = {Bologna, Italy},
}
```
