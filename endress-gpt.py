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

iface = gr.Interface(fn=query_function, inputs="text", outputs="text", title="Chatbot Endress&Hauser")


iface.launch()

# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot()
#     msg = gr.Textbox()
#     clear = gr.Button("Clear")
#     chat_history = []

#     def user(user_message, history):
#         print("User message:", user_message)
#         print("Chat history:", history)

#         response = qa({"question": user_message, "chat_history": history})
#         history.append((user_message, response["answer"]))
#         print("Updated chat history:", history)
#         return gr.update(value=""), history

# msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)

# clear.click(lambda: None, None, chatbot, queue=False)

# if __name__ == "__main__":
#     demo.launch(debug=True)


# def ask_bot(text):
#     output = 