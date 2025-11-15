# install FlagEmbedding!
# pip install FlagEmbedding

import pandas as pd
from tqdm import tqdm
from FlagEmbedding import FlagReranker

# import time
# start = time.time()

# ---------------------------
# 1. Загрузка модели
# ---------------------------
# model_name = "BAAI/bge-reranker-v2-m3"
model = FlagReranker(
    'BAAI/bge-reranker-v2-m3',
    use_fp16=True  # ускоряет на GPU
)

# ---------------------------
# 2. Загружаем данные
# ---------------------------
# q_rrf: q_id, web_id, rrf_score
# questions: q_id, query_clean -- min preprocessed!
# websites: web_id, title_clean, text_clean -- min preprocessed!

base_url = "https://raw.githubusercontent.com/p0velentius/rug-pull/main/"

q_rrf = pd.read_csv(base_url + "model/rrf_results.csv")
questions = pd.read_csv(base_url + "preprocessing/questions_min_preprocessed.csv")
websites = pd.read_csv(base_url + "preprocessing/websites_min_preprocessed.csv")
# to test the workability
# q_rrf = q_rrf.iloc[:1000,:]

# limit text length
websites["text_clean"] = websites["text_clean"].str.slice(0, 10_000)

print(q_rrf.shape, questions.shape, websites.shape)

# Делаем merge, чтобы у каждого кандидата был текст сайта и запрос
df = (
    q_rrf
    .merge(questions, on="q_id", how="left")
    .merge(websites, on="web_id", how="left")
)

# Сформируем поле, которое пойдёт в reranker (заголовок усиливаем)
df["document_text"] = df["title_clean"] + " " + df["text_clean"]

# ---------------------------
# 3. Прогон через reranker
# ---------------------------

# Готовим список пар: (query, document)
# pairs = list(zip(df["query_clean"].tolist(), df["document_text"].tolist()))

# Приводим тексты к строкам + заменяем NaN (там была последняя пустая строка)
df["query_clean"] = df["query_clean"].fillna("").astype(str)
df["document_text"] = df["document_text"].fillna("").astype(str)

# Удаляем строки, где документ пустой (reranker не обработает)
df = df[df["document_text"].str.strip() != ""]

# Формируем пары
pairs = list(zip(df["query_clean"], df["document_text"]))


# Прогоняем батчами
scores = model.compute_score(pairs, batch_size=256)

df["rerank_score"] = scores

# ---------------------------
# 4. Выбираем TOP-5 для каждого запроса
# ---------------------------

# сортируем по q_id → по убыванию rerank_score
df_sorted = df.sort_values(["q_id", "rerank_score"], ascending=[True, False])

# оставляем top5
top5 = df_sorted.groupby("q_id").head(5)

# финальный вывод: q_id, web_id_top5, score_top5
q_reranker = top5[["q_id", "web_id", "rerank_score"]].rename(
    columns={
        "web_id": "web_id_top5",
        "rerank_score": "score_top5"
    }
).reset_index(drop=True)

# ---------------------------
# 5. Сохраняем
# ---------------------------
q_reranker.to_csv("q_reranker_top5.csv", index=False)

# end = time.time()
# elapsed_time = end - start
# print(f"Done in {elapsed_time:.2f} seconds")

print(q_reranker.head())

# q_reranker: q_id, web_id_top5, score_top5

# Группируем web_id_top5 в список
q_to_web_list = (
    q_reranker
    .groupby("q_id")["web_id_top5"]
    .apply(list)
    .reset_index()
    .rename(columns={"web_id_top5": "web_list"})
)

# file for submission
q_to_web_list.to_csv("reranker_result.csv", index=False)

print(q_to_web_list.head())
