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
from rank_bm25 import BM25Okapi

morph = MorphAnalyzer()

def lemmatize_ru(text: str) -> list[str]:
    lemmas = []
    for t in tokenize(text):               # корректная русская токенизация
        token = t.text.lower().replace('ё','е')
        if token.isalpha():                # фильтр пунктуации/чисел
            lemmas.append(morph.parse(token)[0].normal_form)
    return lemmas

morph = MorphAnalyzer()

def preprocess(text: str) -> List[str]:
    # токенизация + нормализация + фильтр пунктуации/чисел
    out = []
    for t in tokenize(text):
        tok = t.text.lower().replace('ё', 'е')
        if tok.isalpha():
            out.append(morph.parse(tok)[0].normal_form)
    return out

# Препроцессим
tokenized_corpus = [preprocess(d) for d in docs]
bm25 = BM25Okapi(tokenized_corpus)  # можно BM25L/BM25Plus
def search(query: str, topk: int = 3):
    q = preprocess(query)
    scores = bm25.get_scores(q)              # список скорингов по документам
    print(scores)
    # берём индексы топ-k
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
    return [(i, scores[i], docs[i]) for i in top_idx]

base_url = "https://raw.githubusercontent.com/p0velentius/rug-pull/main/"

questions = pd.read_csv(base_url + "questions_preprocessed.csv")

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
