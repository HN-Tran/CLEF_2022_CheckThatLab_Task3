import torch
import gc
import re
import emoji
import numpy as np
import pandas as pd
from datasets import load_dataset
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'F1': f1_score(labels, predictions, average=None), 
            'Macro F1': f1_score(labels, predictions, average='macro'), 
            'Accuracy': accuracy_score(labels, predictions),
            'MCC': matthews_corrcoef(labels, predictions)}

# Inference
def inference():
    counter = 0
    for each in CHOSEN:
        counter += 1
        tokenizer = AutoTokenizer.from_pretrained(each)
        model = AutoModelForSequenceClassification.from_pretrained(each, local_files_only=True, num_labels=4).to("cuda")
        test_dataset = load_dataset('csv', data_files={'test': 'test.csv'})
        test_dataset = test_dataset.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=256), batched=True)
        args = TrainingArguments(per_device_eval_batch_size = 32, output_dir = 'tmp_trainer', seed=3)
        trainer = Trainer(model=model,
                        args=args,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics)

        # Predict test dataset
        print(50*"-" + "\n\n#" + str(counter) + " Inference result:\n")
        print(trainer.predict(test_dataset['test']).metrics)

if __name__ == "__main__":
    CHOSEN_MODEL_TYPE = int(input("Which summarization model type did you use?\n\n[1] DistilBART-CNN-12-6\n[2] T5-3B\n\nSelection: "))
    CHOSEN = []
    for x in range(5):
        CHOSEN.append(input("Please type in the PATH to the " + str(x+1) + ". fine-tuned model run folder: "))

    test_df = pd.read_csv(input("Please type in the PATH of the CSV file for inference/prediction: "))
    if CHOSEN_MODEL_TYPE == 1:
        en_test = test_df[['distilbart_cnn_summary_clean','our rating']]
    else:
        en_test = test_df[['T5_summary_clean','our rating']]
    en_test = en_test.dropna()
    en_test['our rating'] = en_test['our rating'].replace({"FALSE": 0, "false":0, "False":0, "true":1, "True":1, "TRUE": 1, "Other": 2, "other": 2, "partially false": 3, "Partially False": 3, "Partially false": 3})
    en_test = en_test.astype({"our rating": int})
    en_test.columns = ["text", "label"]
    en_test['label'] = en_test['label'].apply(lambda x: str(x))
    test = en_test
    test.to_csv('test.csv', index=False)

    inference()
    model = None
    trainer = None
    torch.cuda.empty_cache()
    gc.collect()
