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
# os.environ["OPENAI_API_KEY"] = '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'

# インターネット上からマニュアルファイルをダウンロード（BuffaloTerastation)
url = 'https://manual.buffalo.jp/buf-doc/35021178-39.pdf'
response = requests.get(url)
# Ensure that the request was successful
if response.status_code == 200:
    # Save the content of the response as a PDF file
    with open('sample_document1.pdf', 'wb') as file:
        file.write(response.content)
else:
    print("Error: Unable to download the PDF file. Status code:", response.status_code)


# ページごとに分割。この方法だと、メタデータの情報が取得できるので、マニュアルのページ数などを表示することも可能となるが、
# トークンサイズが大きくなりがち。
# また、PDFの様なページ分割されている情報がソースとなっている必要がある
# Simple method - Split by pages 
# https://manual.buffalo.jp/buf-doc/35021178-39.pdf
#loader = PyPDFLoader("/content/sample_data/buffalo_manual.pdf")
loader = PyPDFLoader("sample_document1.pdf")
pages = loader.load_and_split()
print(pages[3])

# SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
chunks = pages

# Get embedding model
embeddings = OpenAIEmbeddings()

#  vector databaseの作成
db = FAISS.from_documents(chunks, embeddings)
query = "ランプが点滅しているが、これは何が原因か？"
# FAISSに対して検索。検索は文字一致ではなく意味一致で検索する(Vector, Embbeding)
docs = db.similarity_search(query)
# 得られた情報から回答を導き出すためのプロセスを以下の4つから選択する。いずれもProsとConsがあるため、適切なものを選択する必要がある。
# staffing ... 得られた候補をそのままインプットとする
# map_reduce ... 得られた候補のサマリをそれぞれ生成し、そのサマリのサマリを作ってインプットとする
# map_rerank ... 得られた候補にそれぞれスコアを振って、いちばん高いものをインプットとして回答を得る
# refine  ... 得られた候補のサマリを生成し、次にそのサマリと次の候補の様裏を作ることを繰り返す
chain = load_qa_chain(OpenAI(temperature=0.1,max_tokens=1000), chain_type="stuff")
# 得られた情報から回答を導き出すためのプロセスを以下の4つから選択する。いずれもProsとConsがあるため、適切なものを選択する必要がある。
query = "バックアップにはどの様な方法がありますか？またその手順について詳しくおしえてください"
docs = db.similarity_search(query)
# langchainを使って検索
result = chain.run(input_documents=docs, question=query)
print("\n\n" + result)