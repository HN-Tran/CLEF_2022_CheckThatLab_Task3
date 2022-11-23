#!pip install -qqq transformers simpletransformers emoji sentencepiece nltk --upgrade

import logging
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import emoji
import gc
import torch

MODEL=('t5-3b', 'en')

# Seed choice
SEEDS=[2]

# load dataset
df = pd.read_csv(input("Please type in the PATH of the training CSV file: "))
dev_df = pd.read_csv(input("Please type in the PATH of the development CSV file: "))
CHOSEN_MODEL_TYPE = int(input("Which summarization model type did you use?\n\n[1] DistilBART-CNN-12-6\n[2] T5-3B\n\nSelection: "))
if CHOSEN_MODEL_TYPE == 1:
    en_data = df[['distilbart_cnn_summary','our rating']]
    en_dev = dev_df[['distilbart_cnn_summary','our rating']]
else:
    en_data = df[['T5_summary_clean','our rating']]
    en_dev = dev_df[['T5_summary_clean','our rating']]
en_data = en_data.dropna()
en_dev = en_dev.dropna()
en_data['our rating'] = en_data['our rating'].replace({"FALSE": 0, "false":0, "true":1, "TRUE": 1,"other": 2, "partially false": 3})
en_dev['our rating'] = en_dev['our rating'].replace({"FALSE": 0, "false":0, "true":1, "TRUE": 1,"other": 2, "partially false": 3}) 
en_data = en_data.astype({"our rating": int})
en_dev = en_dev.astype({"our rating": int})
en_data.columns = ["text", "label"]
en_dev.columns = ["text", "label"]

lst = []
test_lst = []
for each in en_data.iterrows():
    lst.append("Multiclass Classification")
en_data['prefix'] = lst
en_data = en_data[['prefix', 'text', 'label']]
for each in en_dev.iterrows():
    test_lst.append("Multiclass Classification")
en_dev['prefix'] = test_lst
en_dev = en_dev[['prefix', 'text', 'label']]
    
GRAD_ACC = int(input("\n"+50*"-"+"\n\nGradient Accumulation?\n\n[1] No\n[2] Yes (GA = 2)\nSelection: "))
BATCH_SIZE = int(input("\n"+50*"-"+"\n\nType in batch size per GPU unit: "))
MAX_LENGTH = int(input("\n"+50*"-"+"\n\nType in maximum input length for model: "))
print("\n"+50*"-"+"\n")

def macro_f1_metric(true, predicted):
    return f1_score(true, predicted, average='macro')

def accuracy_metric(true, predicted):
    return accuracy_score(true, predicted)
    
def finetune(data, dev, seed):
    dev, test = train_test_split(dev, test_size=0.2, random_state=101, stratify=dev['label'])
    data.columns = ["prefix", "input_text", "target_text"]
    dev.columns = ["prefix", "input_text", "target_text"]
    test.columns = ["prefix", "input_text", "target_text"]
    data['target_text'] = data['target_text'].apply(lambda x: str(x))
    dev['target_text'] = dev['target_text'].apply(lambda x: str(x))
    test['target_text'] = test['target_text'].apply(lambda x: str(x))
    test.to_csv('test_t5.csv')
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    model_args = T5Args()
    model_args.num_train_epochs = 200
    model_args.no_save = False
    model_args.overwrite_output_dir = True
    model_args.evaluate_generated_text = True
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_verbose = True
    model_args.n_gpu = 1
    model_args.fp16 = False
    model_args.max_seq_length = MAX_LENGTH
    model_args.use_early_stopping = True
    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_metric = "macro_f1"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_consider_epochs = True
    model_args.early_stopping_patience = 5
    model_args.gradient_accumulation_steps = GRAD_ACC
    model_args.train_batch_size = BATCH_SIZE
    model_args.eval_batch_size = BATCH_SIZE
    model_args.manual_seed = seed
    
    model = T5Model("t5", MODEL[0], args=model_args, use_cuda=True)

    def count_matches(labels, preds):
        return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])

    model.train_model(data, eval_data=dev, matches=count_matches, macro_f1=macro_f1_metric, accuracy=accuracy_metric)
    print(model.eval_model(test, matches=count_matches, macro_f1=macro_f1_metric, accuracy=accuracy_metric))

if __name__ == "__main__":
    for seed in SEEDS:
        finetune(en_data, en_dev, seed)
        model = None
        torch.cuda.empty_cache()
        gc.collect()