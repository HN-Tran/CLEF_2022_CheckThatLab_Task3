import logging
import torch
import gc
import re
import emoji
import numpy as np
import pandas as pd
from datasets import load_dataset
#from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from simpletransformers.t5 import T5Model, T5Args

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'Macro F1': f1_score(labels, predictions, average='macro'), 
            'Accuracy': accuracy_score(labels, predictions),
            'Precision': precision_score(labels, predictions), 
            'Recall': recall_score(labels, predictions)}

def macro_f1_metric(true, predicted):
    return f1_score(true, predicted, average='macro')

def inference(test_lst, test):
    CHOSEN = input("Please type in the PATH to the fine-tuned T5 model folder: ")
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    model_args = T5Args()
    model_args.no_save = False
    model_args.overwrite_output_dir = True
    model_args.n_gpu = 1
    model_args.fp16 = False
    model_args.max_seq_length = 256
    model_args.eval_batch_size = 32
    model = T5Model("t5", CHOSEN, args=model_args, use_cuda=True)
    print(50*"-" + "\n\n# Predict Dataset:\n")
    pred = model.predict(test_lst)
    print(macro_f1_metric(test['label'].tolist(), pred))
    test['predicted_rating'] = pred
    test['predicted_rating'] = test['predicted_rating'].replace({"0":"false", "1":"true", "2": "other", "3": "partially false"})
    test = test[['ID', 'predicted_rating']]
    test.columns = ['public_id', 'predicted_rating']
    #test.to_csv("tsv/abstractive-t5-en.tsv", sep="\t", index=False)

if __name__ == "__main__":
    lst_train = []
    lst_test = []
    temp = []
    lst_p_train = []
    lst_p_test = []
    temp_p = []
    test_df = pd.read_csv(input("Please type in the PATH of the CSV file for inference/prediction: "))
    en_test = test_df[['T5_summary_clean','our rating']]
    en_test = en_test.dropna()
    en_test['our rating'] = en_test['our rating'].replace({"FALSE": 0, "false":0, "False":0, "true":1, "True":1, "TRUE": 1, "Other": 2, "other": 2, "partially false": 3, "Partially False": 3, "Partially false": 3})
    en_test = en_test.astype({"our rating": int})
    en_test.columns = ["text", "label"]
    en_test['text'] = "Multiclass Classification: " + en_test['text']
    en_test['label'] = en_test['label'].apply(lambda x: str(x))
    test = en_test
    #en_test = test_df.merge(test_df2, how='inner', on='ID')
    test.to_csv('test.csv', index=False)
    #dev, test = train_test_split(en_dev, test_size=0.2, random_state=101, stratify=en_dev['label'])
    #test.columns = ["prefix", "input_text", "target_text"]
    ##train['target_text'] = train['target_text'].apply(lambda x: str(x))
    #test['target_text'] = test['target_text'].apply(lambda x: str(x))
    #test['input_text'] = "Multiclass Classification: " + test['input_text']
    test_lst = []
    for each in test['text']:
    #for each in test['input_text']:
        test_lst.append(each)
    inference(test_lst, test)
    model = None
    trainer = None
    torch.cuda.empty_cache()
    gc.collect()
