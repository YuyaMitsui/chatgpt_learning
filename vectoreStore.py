import os
from dotenv import load_dotenv
import pandas as pd
import requests
import textract
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain


# .envファイルの内容を読み込見込む
load_dotenv()

with open('recruit.txt', 'r') as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    # chunk_size = 512,
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

# Get embedding model
embeddings = OpenAIEmbeddings()

#  vector databaseの作成
db = FAISS.from_documents(chunks, embeddings)
query = "話すことが好きな人が向いている企業は"
# FAISSに対して検索。検索は文字一致ではなく意味一致で検索する(Vector, Embbeding)
docs = db.similarity_search(query)
# 得られた情報から回答を導き出すためのプロセスを以下の4つから選択する。いずれもProsとConsがあるため、適切なものを選択する必要がある。
# staffing ... 得られた候補をそのままインプットとする
# map_reduce ... 得られた候補のサマリをそれぞれ生成し、そのサマリのサマリを作ってインプットとする
# map_rerank ... 得られた候補にそれぞれスコアを振って、いちばん高いものをインプットとして回答を得る
# refine  ... 得られた候補のサマリを生成し、次にそのサマリと次の候補の様裏を作ることを繰り返す
chain = load_qa_chain(OpenAI(temperature=0.1,max_tokens=1000), chain_type="stuff")
# 得られた情報から回答を導き出すためのプロセスを以下の4つから選択する。いずれもProsとConsがあるため、適切なものを選択する必要がある。
query = "話すことが好きな人が向いている企業は"
docs = db.similarity_search(query)
# langchainを使って検索
result = chain.run(input_documents=docs, question=query)
print("\n\n" + result)