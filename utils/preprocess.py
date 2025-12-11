import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

with open("./model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)


def clean_text(text):
    delete_dict = {sp_character: '' for sp_character in string.punctuation}
    delete_dict[' '] = ' '
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr= text1.split()
    text2 = ' '.join([w for w in textArr if ( not w.isdigit() and  ( not w.isdigit() and len(w)>3))])
    return text2.lower()

def preprocess(text) :
    text = remove_emoji(text)
    text2 = clean_text(text)
    lst = tokenizer.texts_to_sequences([text2])
    final = pad_sequences(lst, padding='post', maxlen=50)
    return final