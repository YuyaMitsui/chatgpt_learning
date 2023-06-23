import requests
import json
import pandas as pd
from dotenv import load_dotenv
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

# OpenAIのキーを設定します
load_dotenv()
model=OpenAI(model_name="gpt-3.5-turbo",temperature=0)

# 1.求人情報を要約する
# - プロンプト準備
summary_template = PromptTemplate(
    input_variables=["job_data"], 
    template = """
■求人情報がどういったタイプの人物に向いているかを出す。下記は求人情報の文面です。
この求人情報にどのようなタイプの人が向いているかを簡潔に箇条書きで教えてください。

``` 求人情報
{job_data}
```
"""
)
chain1 = LLMChain(
    llm=model, 
    prompt=summary_template, 
    verbose=True, 
    memory=ConversationBufferWindowMemory(k=1),
)

# - CSVファイルを読み込む
csv_data = pd.read_csv('jobdata.csv')
series = csv_data.apply(lambda row: '\n'.join(row.values.astype(str)), axis=1)
result = series.tolist()

# - プロンプト実行
summary_output = chain1.predict(job_data=result[0])
print(summary_output)

# 2.性格とマッチさせる
# - プロンプト準備
matchingTemplate = PromptTemplate(
    input_variables=["combined_data"], 
    template = """
■求人情報の向いているタイプが、どの人物タイプと近いかを出す。下記の特性が求められる求人情報に対して、どの人物タイプが向いていると考えられますか？
それぞれマッチ度を1%刻みで付けたとして、70％以上の人物タイプだけを回答フォーマットに則って教えてください。

# 回答フォーマット
[人物タイプ名]：[パーセンテージ]

{combined_data}
"""
)

chain2 = LLMChain(
    llm=model, 
    prompt=matchingTemplate, 
    verbose=True, 
    memory=ConversationBufferWindowMemory(k=1),
)

# - ファイルの読み込み
with open('personal.txt', 'r') as f:
    personal = f.read()

combined_data = f"""
# ある求人情報が求めている特性
{summary_output}

# 人物タイプ
{personal}
"""
# - プロンプト実行
output = chain2.predict(combined_data=combined_data)
print(output)