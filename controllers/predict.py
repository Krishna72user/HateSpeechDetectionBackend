import pickle
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

with open('./model/tokenizer_v2.pkl','rb') as f:
    tokenizer = pickle.load(f)

model = load_model("./model/hate_speech_modelv2.h5")


def clean_text(text):
  text = text.lower()
  text = re.sub(r"http\S+|www\S+", "", text)
  text = re.sub(r"<.*?>", "", text)
  text = re.sub(r"\d+", "", text)
  text = text.translate(str.maketrans("", "", string.punctuation))
  text = re.sub(r"\s+", " ", text).strip()
  return text

MAX_LEN = 100
def predict_hate(text):
  text = clean_text(text)
  seq = tokenizer.texts_to_sequences([text])
  pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
  prob = model.predict(pad)[0][0]
  return int(prob > 0.5)