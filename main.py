import os
import random
import threading
import time
import pandas as pd
import gradio as gr

print(gr.__version__)

from txtai.embeddings import Embeddings

# 导入会话数据
df = pd.read_csv("./dataset_zh-CN.csv")
print(df)

df2 = df.drop_duplicates(subset="quesion", inplace=False)
col = df2["quesion"]
# print(col)

# for uid, text in enumerate(col):
#     print(uid, text)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 创建 embeddings 模型 (sentence-transformers & transformers)
# embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2")
# embeddings = Embeddings(path="DMetaSoul/sbert-chinese-general-v2")
embeddings = Embeddings(path="DMetaSoul/sbert-chinese-general-v2-distill")

for uid, text in enumerate(col):
    print(uid, text)

# Index the list of text
embeddings.index([(text, text, text) for (uid, text) in enumerate(col)])

# def print_numbers():
#     for i in range(5):
#         time.sleep(1)
#         print(i)


# # 创建线程
# thread = threading.Thread(target=print_numbers)

# # 启动线程
# thread.start()

while True:
    line = input("ME: ").strip()

    if "q" == line:
        break

    res = embeddings.search(line, 1)

    if len(res) > 0:
        (quesion, score) = res[0]
        print(f"{quesion} {score}")

        ds = df.loc[(df["quesion"] == quesion), :]
        rand = random.randint(0, len(ds) - 1)
        answer = ds.iloc[rand]

        print(answer)
    else:
        print("啊")
