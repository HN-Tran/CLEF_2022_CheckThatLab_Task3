import pandas as pd
import emoji
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

def remove_in_word_whitespaces(comment):
    find = re.findall(r'(^| )(([a-zA-Z] ){1,}[a-zA-Z!?,.]([^a-zA-Z]|$))', comment)
    if len(find) > 0:
        for match in find:
            found = match[0] + match[1]
            replacement = ' ' + re.sub(r' ', '', found) + ' '
            comment = comment.replace(found, replacement, 1)
    return comment

def demojize(comment):
    return emoji.demojize(comment, delimiters=(' <', '> '))

def clean_up_translated_comments(df, column):
    df[column] = df[column].apply(lambda t: str(t))
    df[column] = df[column].apply(lambda t: demojize(t))
    df[column] = df[column].apply(lambda t: emoji.emojize(t, delimiters=('<', '>')))
    df[column] = df[column].apply(lambda t: remove_in_word_whitespaces(t))
    df[column] = df[column].str.replace(r' {2,}', ' ', regex=True)
    df[column] = df[column].str.strip()
    return df

def summarizer(dataset, column1, column2):
    for index, each in dataset.iterrows():
        text = str(each[column1]).lower()
        title = str(each[column2]).lower()
        preprocess_text = text.strip().replace("\n","")
        preprocess_title = title.strip().replace("\n","")
        if preprocess_title == '' or pd.isna(preprocess_title):
            t5_prepared_text = "summarize: " + preprocess_text
        elif preprocess_text == '' or pd.isna(preprocess_text):
            t5_prepared_text = "summarize: " + preprocess_title
        else:
            t5_prepared_text = "summarize: " + preprocess_title + ". " + preprocess_text
        print ("original text preprocessed: \n", t5_prepared_text)
        tokenized_text = tokenizer.encode(t5_prepared_text, return_tensors="pt", max_length=3000).to(device)
        summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=512,
                                        early_stopping=True)
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print ("\n\nSummarized text: \n", output)
        output_list.append(output)
        torch.cuda.empty_cache()
    dataset['T5_summary_clean'] = output_list
    dataset.to_csv(input("Please name your output CSV file without file extension: ") + ".csv", index=False, header=True, encoding="utf-8")
    return dataset

if __name__ == "__main__":
    model = T5ForConditionalGeneration.from_pretrained('t5-3b')
    tokenizer = T5Tokenizer.from_pretrained('t5-3b')
    device = torch.device('cuda:0')
    model.to(device)
    output_list = []
    dataset = pd.read_csv(input("Please type in the PATH of the CSV file with file extension for summarization with T5-3B: "))
    clean_up_translated_comments(dataset, "English_Google_text")
    clean_up_translated_comments(dataset, "English_Google_title")
    summarizer(dataset, "English_Google_text", "English_Google_title")