import os
import random
import time
import gradio as gr
import queue
import threading
import pandas as pd

from txtai.embeddings import Embeddings

# 导入会话数据
df = pd.read_csv("./dataset_zh-CN.csv")
print(df)

df2 = df.drop_duplicates(subset="quesion", inplace=False)
col = df2["quesion"]

os.environ["OMP_NUM_THREADS"] = "1"

# 创建 embeddings 模型 (sentence-transformers & transformers)
# embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2")
# embeddings = Embeddings(path="DMetaSoul/sbert-chinese-general-v2")
embeddings = Embeddings(path="DMetaSoul/sbert-chinese-general-v2-distill")

for uid, text in enumerate(col):
    print(uid, text)

# Index the list of text
embeddings.index([(text, text, text) for (uid, text) in enumerate(col)])


class AppState:
    def __init__(self):
        self.history = [(None, "A")]

    def __str__(self):
        return "{ history: {} }".format(self.history)


def on_chat(question, history, state: AppState):
    state.history.append([question, None])
    return "", state.history


# 每秒更新一次
def on_tick(history, state: AppState):
    print(state)
    state.history.append([None, random.choice(["Great", "Good", "Okay", "Bad"])])
    return state.history


def main():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        state = gr.State(AppState())

        gr.Markdown("限制场景会话")

        with gr.Column():
            chatbot = gr.Chatbot(label="会话列表", bubble_full_width=False)
            question = gr.Textbox(label="用户输入")
            submit = gr.Button("提交")
            with gr.Row():
                pussy = gr.Button("生殖器")
                hands = gr.Button("手")
                legs = gr.Button("腿")
                feet = gr.Button("脚")

        submit.click(
            fn=on_chat,
            inputs=[question, chatbot, state],
            outputs=[question, chatbot],
        )

        demo.load(fn=on_tick, inputs=[chatbot, state], outputs=[chatbot], every=1)

    demo.queue().launch()


if __name__ == "__main__":
    main()
