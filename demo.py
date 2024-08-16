import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI



_=load_dotenv(find_dotenv())

os.environ['HTTPS_PROXY']='http://127.0.0.1:7890'
os.environ['HTTP_PROXY']='http://127.0.0.1:7890'



client = OpenAI(
    # This is the default and can be omitted
    api_key='',
)

# 导入所需库
# 注意，此处我们假设你已根据上文配置了 OpenAI API Key，如没有将访问失败
completion = client.chat.completions.create(
    # 调用模型：ChatGPT-3.5
    model="gpt-3.5-turbo",
    # messages 是对话列表
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
print(completion.choices[0].message.content)