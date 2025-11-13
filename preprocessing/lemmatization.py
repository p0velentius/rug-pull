!pip install rank-bm25 razdel pymorphy3

from typing import List
from razdel import tokenize
from pymorphy3 import MorphAnalyzer
# from rank import BM25Okapi
import pandas as pd
from tqdm.auto import tqdm
from razdel import tokenize
from pymorphy3 import MorphAnalyzer
from typing import List
from razdel import tokenize
from pymorphy3 import MorphAnalyzer

morph = MorphAnalyzer()

def lemmatize_ru(text: str) -> list[str]:
    lemmas = []
    for t in tokenize(text):               # корректная русская токенизация
        token = t.text.lower().replace('ё','е')
        if token.isalpha():                # фильтр пунктуации/чисел
            lemmas.append(morph.parse(token)[0].normal_form)
    return lemmas

morph = MorphAnalyzer()

base_url = "https://raw.githubusercontent.com/p0velentius/rug-pull/main/"

questions = pd.read_csv(base_url + "questions_preprocessed.csv", index_col=0)

QUESTIONS_TEXT_COL = "query_clean"

tqdm.pandas()

def lemmatize_text_series(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna('')
        .astype(str)
        .progress_apply(lambda txt: ' '.join(lemmatize_ru(txt)))
    )

questions["lemmas"] = lemmatize_text_series(questions[QUESTIONS_TEXT_COL])


questions['query_clean'].reset_index().to_csv('questions_lemmatizated.csv', index=False)

print("Файл предобработан и сохранен в questions_lemmatizated.csv")
