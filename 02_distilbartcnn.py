import pandas as pd
import emoji
import re
from transformers import AutoConfig, AutoTokenizer, AutoModel
from summarizer import Summarizer

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

def summarizer(dataset):
    for index,each in dataset.iterrows():
        text = str(each['English_Google_text'])
        title = str(each['English_Google_title'])
        preprocess_text = text.strip().replace("\n", "")
        preprocess_title = title.strip().replace("\n", "")
        if preprocess_title == '' or pd.isna(preprocess_title):
           body = preprocess_text
        elif preprocess_text == '' or pd.isna(preprocess_text):
            body = preprocess_title
        else:
            body = preprocess_title + ". " + preprocess_text
        print ("\noriginal text preprocessed: \n\n", body)
        result = model(body, num_sentences=model.calculate_optimal_k(body, k_max=10))
        print("-" * 100)
        print ("\nsummarized text output: \n\n", result)
        print("=" * 100)
        output_list.append(result)
    dataset['distilbart_cnn_summary'] = output_list
    dataset.to_csv(input("Please name your output CSV file without file extension: ") + ".csv", index=False, header=True, encoding="utf-8")
    return dataset

if __name__ == "__main__":
    model_path = 'sshleifer/distilbart-cnn-12-6'
    custom_config = AutoConfig.from_pretrained(model_path)
    custom_config.output_hidden_states=True
    custom_tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
    custom_model = AutoModel.from_pretrained(model_path, config=custom_config)
    model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
    output_list = []
    dataset = pd.read_csv(input("Please type in the PATH of the CSV file with file extension for summarization with DistilBART-CNN-12-6: "))
    clean_up_translated_comments(dataset, "English_Google_text")
    clean_up_translated_comments(dataset, "English_Google_title")
    summarizer(dataset)
