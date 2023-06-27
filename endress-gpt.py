import os
import sys

from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

import gradio as gr

import constants

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

# query = sys.argv[1]

loader = TextLoader('data.txt')
index = VectorstoreIndexCreator().from_loaders([loader])

# print(index.query(query, llm=ChatOpenAI()))

def query_function(query):
    loader = TextLoader('data.txt')
    index = VectorstoreIndexCreator().from_loaders([loader])
    return index.query(query, llm=ChatOpenAI())

def clear():
    return None, None

# iface = gr.Interface(fn=query_function, inputs="text", outputs="text", title="Chatbot Endress&Hauser")

with gr.Blocks() as iface:
    gr.Markdown(
        """
        # Endress & Hauser GPT
        Stelle deine Frage und erhalte eine Antwort.
        """)
    with gr.Row():
        inp = gr.Textbox(label="Deine Frage:", placeholder="Was möchtest du wissen?")
        out = gr.Textbox(label="Die Antwort:")
    btn = gr.Button("Los geht's!")
    btn.click(fn=query_function, inputs=inp, outputs=out)
    clear = gr.ClearButton()


if __name__ == "__main__":
    iface.launch()