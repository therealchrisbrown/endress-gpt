import os
import sys

from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

import constants

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

query = sys.argv[1]

loader = TextLoader('data.txt')
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query, llm=ChatOpenAI()))