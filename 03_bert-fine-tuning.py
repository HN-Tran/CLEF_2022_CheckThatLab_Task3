# !pip install -qqq torch transformers datasets==1.5.0 emoji sentencepiece --upgrade

import gc
import torch
import emoji
import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments 

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'Macro F1': f1_score(labels, predictions, average='macro'),
            'Accuracy': accuracy_score(labels, predictions),
            'MCC': matthews_corrcoef(labels, predictions)}

def finetune_data(data, dev, seed):
    #train, dev = train_test_split(data, test_size=0.2, random_state=101, stratify=data['class_label'])
    data.columns = ["text", "label"]
    dev.columns = ["text", "label"]
    dev, test = train_test_split(dev, test_size=0.2, random_state=101, stratify=dev['label'])
    data.to_csv('train.csv', index=False)
    dev.to_csv('dev.csv', index=False)
    test.to_csv('dev_test.csv', index=False)
    dataset = load_dataset('csv', data_files={'train': 'train.csv',
                                              'dev': 'dev.csv',
                                              'test': 'dev_test.csv'})
    label_list = dataset['train'].unique('label')
    label_list.sort()
    num_labels = len(label_list)
    tokenizer = AutoTokenizer.from_pretrained(MODEL[0])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL[0], num_labels=num_labels)
    dataset = dataset.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=MAX_LENGTH), batched=True)
    args = TrainingArguments(
      output_dir = 'output',
      overwrite_output_dir = True,
      num_train_epochs = 5,
      evaluation_strategy = "steps",
      save_strategy = "steps",
      save_steps = 50,
      learning_rate = LEARNING_RATE,       
      weight_decay = 0.01, 
      logging_steps = 50,                    
      load_best_model_at_end = True,
      metric_for_best_model = 'Macro F1',
      per_device_train_batch_size = BATCH_SIZE,
      per_device_eval_batch_size = BATCH_SIZE*GRAD_ACC,
      run_name = 'Seed: ' + str(seed),
      seed = seed,
      warmup_ratio = 0.1,
      gradient_accumulation_steps = GRAD_ACC,
      max_steps = 705
    )
    trainer = Trainer(
      model = model,
      args = args,
      train_dataset=dataset['train'],
      eval_dataset=dataset['dev'],
      tokenizer=tokenizer,
      compute_metrics=compute_metrics
    )
    model.eval()
    trainer.train()
    name = re.sub(r'\/', r'_', str(MODEL[0]))
    trainer.save_model("best_models/" + name + "_" + str(seed)) 
    return trainer

if __name__ == "__main__":
    MODEL_CHOICE = []
    TRANSLATOR = 0

    # Model and language choice
    MODELS_BASE=[('bert-base-uncased', 'en', 1e-5),
                ('bert-large-uncased', 'en', 1e-5),
                ('xlm-roberta-large', 'en', 1e-5)]

    # Seed choice
    SEEDS=[2,3,5,7,11]

    # Task and model user interaction
    MODEL_NUM = int(input("\n"+50*"-"+"\n\n"+"""\
    Choose model:
    [1] BERT Uncased Base
    [2] BERT Uncased Large
    [3] XLM-R Large
    """ + "\nSelection: "))-1
    MODEL_CHOICE.append(MODELS_BASE[MODEL_NUM])
    MODEL = MODEL_CHOICE[0]

    # load dataset
    df = pd.read_csv(input("Please type in the PATH of the training set CSV file: "))
    dev_df = pd.read_csv(input("Please type in the PATH of the development set CSV file: "))

    LEARNING_RATE = MODEL[2]
    GRAD_ACC = int(input("\n"+50*"-"+"\n\nGradient Accumulation?\n\n[1] No\nOr enter your GA step value\nSelection: "))
    BATCH_SIZE = int(input("\n"+50*"-"+"\n\nType in batch size per GPU unit: "))
    MAX_LENGTH = int(input("\n"+50*"-"+"\n\nType in maximum input length for model: "))
    print("\n"+50*"-"+"\n")

    CHOSEN_MODEL_TYPE = int(input("Which summarization model type did you use?\n\n[1] DistilBART-CNN-12-6\n[2] T5-3B\n\nSelection: "))
    if CHOSEN_MODEL_TYPE == 1:
        en_data = df[['distilbart_cnn_summary','our rating']]
        en_dev = dev_df[['distilbart_cnn_summary','our rating']]
    elif CHOSEN_MODEL_TYPE == 2:
        en_data = df[['T5_summary_clean','our rating']]
        en_dev = dev_df[['T5_summary_clean','our rating']]
    en_data = en_data.dropna()
    en_dev = en_dev.dropna()
    en_data['our rating'] = en_data['our rating'].replace({"FALSE": 0, "false":0, "true":1, "TRUE": 1,"other": 2, "partially false": 3})
    en_dev['our rating'] = en_dev['our rating'].replace({"FALSE": 0, "false":0, "true":1, "TRUE": 1,"other": 2, "partially false": 3}) 
    en_data = en_data.astype({"our rating": int})
    en_dev = en_dev.astype({"our rating": int})

    for seed in SEEDS:
        finetune_data(en_data, en_dev, seed)
        model = None
        trainer = None
        torch.cuda.empty_cache()
        gc.collect()
