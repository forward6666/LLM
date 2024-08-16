import os
from dotenv import load_dotenv,find_dotenv
from langchain_openai import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
import sys
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



load_dotenv(find_dotenv())
# sys.path.append("../C3 搭建知识库")
#导入向量库
embedding=OpenAIEmbeddings()
persist_directory='data_base/vector_db\chroma_1'
vectordb=Chroma(persist_directory=persist_directory,embedding_function=embedding)

#测试
# print(f"向量库中存储的数量：{vectordb._collection.count()}")
# question='什么是prompt engineering?'
# docs=vectordb.similarity_search(question,k=3)
# print(len(docs))
# for i,doc in enumerate(docs):
#     print(f'第{i}个检索答案为：\n{doc.page_content}',end='\n----------------------------------------------------\n')


llm=ChatOpenAI(temperature=0,streaming=True)

template="""使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
"""
QA_CHAIN_PROMPT=PromptTemplate(input_variables=['context','question'],template=template)

qa_chain=RetrievalQA.from_chain_type(llm=llm,retriever=vectordb.as_retriever(),
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt':QA_CHAIN_PROMPT})
#知识库加大模型的回答
# question_1 = "什么是南瓜书？"
# result=qa_chain({"query":question_1})
# print(result['result'])

memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)

qa=ConversationalRetrievalChain.from_llm(llm,retriever=vectordb.as_retriever(),memory=memory)

#测试记忆对话的对话检索链
question='我可以学习到关于提示工程的知识吗？'
result=qa({'question':question})
print(result['answer'])
question='为什么这门课要教这方面的知识？'
result=qa({'question':question})
print(result['answer'])