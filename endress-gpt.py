import os
import sys

from decouple import config

OPENAI_API_KEY = config('OPENAI_API_KEY')

query = sys.argv[1]
print(query)