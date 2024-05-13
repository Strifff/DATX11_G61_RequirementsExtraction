import os
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")


class Tokenizer:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def extract_sentences(self, text):
        sentences = sent_tokenize(text)
        return sentences

    def tokenize_sentences(self, sentences, inc_stopwords=True):
        stop_words = set(stopwords.words("english"))

        words = word_tokenize(sentences)

        if not inc_stopwords:
            # Optionally remove stopwords
            words = [word for word in words if word.lower() not in stop_words]

        return words
