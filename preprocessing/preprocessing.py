import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# download df
base_url = "https://raw.githubusercontent.com/p0velentius/rug-pull/main/"

questions = pd.read_csv(base_url + "questions_clean.csv", index_col=0)

print(questions.shape)

questions.head()
questions.info()

import re
import unicodedata
from typing import List

# ---- конфиги ----
POLITE_PHRASES = [
    r'здравствуйте', r'добрый\s+день', r'добрый\s+вечер', r'доброе\s+утро', r'доброй\s+ночи',
    r'привет(ствую)?', r'добрый', r'добрыйдень', r'хотел(а)', r'подскажите', r'скажите', r'бы',
    r'пожалуйста', r'спасибо', r'заранее', r'прошу', r'извините', r'извините', r'вот', r'пока'
]
# объединяем в одно регулярное выражение (по словам, нечувствительно к регистру)
_POLITE_RE = re.compile(r'\b(' + r'|'.join(POLITE_PHRASES) + r')\b', flags=re.IGNORECASE)

# эмодзи: охватываем общие Unicode блоки (на практике хватает)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)

# шаблон для обнаружения токенов полностью состоящих из 0 или X/x (анонимизированные числа)
_ANON_NUM_RE = re.compile(r'\b[0Xx]+\b')

# URL / EMAIL / @username detection (простые паттерны)
_URL_RE = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', flags=re.IGNORECASE)
_AT_RE = re.compile(r'@\w+')

# пунктуация: удалим все символы категории Unicode "P" (пунктуация),
# но сохраним символы "<" и ">" (если мы вставили токены вида <ANON_NUM>)
def remove_punctuation_keep_tokens(text: str) -> str:
    result = []
    for i, ch in enumerate(text):
        if unicodedata.category(ch).startswith('P') and ch not in '<>':
            # если пунктуация стоит между буквами/цифрами → заменяем на пробел
            if i > 0 and i < len(text) - 1:
                if text[i-1].isalnum() and text[i+1].isalnum():
                    result.append(' ')
                    continue
            # иначе просто пропускаем
            continue
        result.append(ch)
    return ''.join(result)

# небольшие утилиты
def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

# основной препроцессинг одного текста
def preprocess_text(text: str,
                    do_lower: bool = True,
                    replace_anon_with: str = '<ANON_NUM>',
                    remove_emojis: bool = True,
                    remove_punct: bool = True,
                    remove_polite: bool = True,
                    do_lemmatize: bool = False,
                    remove_short_tokens: bool = False,
                    min_token_len: int = 2) -> str:
    if not isinstance(text, str):
        return ''

    s = text

    # 1) убрать URL, email, @username, добавить токены
    s = _URL_RE.sub(' <URL> ', s)
    s = _EMAIL_RE.sub(' <EMAIL> ', s)
    s = _AT_RE.sub(' <USER> ', s)

    # 2) удалить эмодзи
    if remove_emojis:
        s = _EMOJI_RE.sub(' ', s)

    # 3) заменить анонимизированные числа (последовательности 0 или X/x) на единый токен
    s = _ANON_NUM_RE.sub(' ' + replace_anon_with + ' ', s)

    # 4) убрать вежливости / обращения (по списку)
    if remove_polite:
        s = _POLITE_RE.sub(' ', s)

    # 5) to lower
    if do_lower:
        s = s.lower()

    # 6) удалить пунктуацию (сохраняя токены вида <ANON_NUM>)
    if remove_punct:
        s = remove_punctuation_keep_tokens(s)

    # 7) нормализовать пробелы
    s = normalize_whitespace(s)

    # 8) (опция) лемматизация
    if do_lemmatize:
        lemmas = lemmatize_ru(s)
        if remove_short_tokens:
            lemmas = [t for t in lemmas if len(t) >= min_token_len]
        s = ' '.join(lemmas)

    else:
        # удалить короткие токены (если нужно) — после пунктуации и нормализации
        if remove_short_tokens:
            toks = [t for t in s.split() if len(t) >= min_token_len]
            s = ' '.join(toks)

    return s

# удобная функция для DataFrame
def preprocess_df(df: pd.DataFrame,
                  text_col: str = 'query',
                  out_col: str = 'query_clean',
                  inplace: bool = False,
                  **preprocess_kwargs) -> pd.DataFrame:
    """
    Применяет preprocess_text ко всем записям колонки text_col.
    Возвращает DataFrame с новой колонкой out_col (если inplace==False) либо обновляет DataFrame.
    preprocess_kwargs передаются в preprocess_text.
    """
    if text_col not in df.columns:
        raise ValueError(f'Колонка {text_col} не найдена в DataFrame')

    target = df if inplace else df.copy()
    target[out_col] = target[text_col].fillna('').astype(str).apply(lambda t: preprocess_text(t, **preprocess_kwargs))
    return target

# стандартная обработка: lowercase, удалить эмодзи, токен <ANON_NUM>, убрать вежливости
df_clean = preprocess_df(questions,
                         text_col='query',
                         out_col='query_clean',
                         inplace=False,
                         do_lower=True,
                         remove_emojis=True,
                         remove_punct=True,
                         remove_polite=True,
                         do_lemmatize=False,      # поставьте True если установили pymorphy2 и razdel
                         remove_short_tokens=True)

# Установка опции для отображения всей ширины столбца
pd.set_option('display.max_colwidth', None)

print(df_clean[['query', 'query_clean']])

df_clean['query_clean'].reset_index().to_csv('questions_preprocessed.csv', index=False)

print("Файл предобработан и сохранен в questions_preprocessed.csv")
