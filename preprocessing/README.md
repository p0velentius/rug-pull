Preprocessing data

df_clean = preprocess_df(questions,
                         text_col='query',
                         out_col='query_clean',
                         inplace=False,
                         do_lower=True,
                         remove_emojis=True,
                         remove_punct=True,
                         remove_polite=True,
                         do_lemmatize=False,
                         remove_short_tokens=True)

preprocessing.py
input: questions_clean.csv (6977,  2)
output: questions_preprocessed.csv (6977,  2)
< 1 sec.

Added questions_preprocessed.csv:
- q_id
- query_clean
Коротко о решениях:
- Анонимные числа (0000, XXXX, 0, XX и т.п.) заменяем на <ANON_NUM> — чтобы векторизатор видел один токен вместо множества вариантов.
- Эмодзи удаляю регуляркой по Unicode-диапазонам (надёжнее, чем вручную).
- Пунктуация удаляется аккуратно (оставляем только слова и специальные токены вроде <ANON_NUM>).
