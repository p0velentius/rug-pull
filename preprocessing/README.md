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
