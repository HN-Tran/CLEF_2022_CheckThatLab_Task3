import pandas as pd
import emoji
import re

from deep_translator import GoogleTranslator

def load_dataset():
    df = pd.read_csv(input("[IMPORTANT] Your CSV file should contain these two columns: 'title', 'text'\nPlease type in the PATH of the CSV file with file extension for translation: "))
    clean_up_comments(df)
    return df

def remove_in_word_whitespaces(comment):
    find = re.findall(r'(^| )(([a-zA-ZÄÖÜäöüß] ){1,}[a-zA-ZÄÖÜäöüß!?,.]([^a-zA-ZÄÖÜäöüß]|$))', comment)
    if len(find) > 0:
        for match in find:
            found = match[0] + match[1]
            replacement = ' ' + re.sub(r' ', '', found) + ' '
            comment = comment.replace(found, replacement, 1)
    return comment

def demojize(comment):
    return emoji.demojize(comment, delimiters=(' <', '> '))

def clean_up_comments(df):
    df['text'] = df['text'].apply(lambda t: demojize(t))
    df['text'] = df['text'].apply(lambda t: emoji.emojize(t, delimiters=('<', '>')))
    df['text'] = df['text'].apply(lambda t: remove_in_word_whitespaces(t))
    df['text'] = df['text'].str.replace(r' {2,}', ' ', regex=True)
    df['text'] = df['text'].str.strip()
    df['title'] = df['title'].apply(lambda t: demojize(t))
    df['title'] = df['title'].apply(lambda t: emoji.emojize(t, delimiters=('<', '>')))
    df['title'] = df['title'].apply(lambda t: remove_in_word_whitespaces(t))
    df['title'] = df['title'].str.replace(r' {2,}', ' ', regex=True)
    df['title'] = df['title'].str.strip()
    return df

def translate_Google_Translate(dataset, lang):
    English_Google_title = GoogleTranslator(source=lang, target='en').translate_batch(dataset['title'].str[:4999].values.tolist())
    English_Google_text = GoogleTranslator(source=lang, target='en').translate_batch(dataset['text'].str[:4999].values.tolist())
    dataset['English_Google_text'] = English_Google_text
    dataset['English_Google_title'] = English_Google_title
    dataset.to_csv(input("Please name your output CSV file without file extension: ") + ".csv", index=False)
    return dataset

def do_translate_all(lang):
    df = load_dataset()
    translate_Google_Translate(df, lang)
    print("Finished translating to " + str(lang).upper())

if __name__ == "__main__":
    do_translate_all(input("The language of the input dataset is (e.g., German): ").lower())