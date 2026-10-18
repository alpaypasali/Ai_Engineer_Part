import re
import pandas as pd


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"\bx{2,}\b", " ", text)

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text



def clean_texts(texts):
    return [clean_text(text) for text in texts]

