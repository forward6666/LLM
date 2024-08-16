import os
from dotenv import load_dotenv,find_dotenv

# 配置文件路径列表
_=load_dotenv(find_dotenv())

file_paths=[]
folder_path ="data_base\knowledge_db"
for root,dirs,files in os.walk(folder_path):
    # print(root)
    # print(dirs)
    # print(files)
    for file in files:
        file_path=os.path.join(root,file)
        file_paths.append(file_path)
print(file_paths)

# 导入知识库
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader

loaders=[]
for file_path in file_paths:
    type=file_path.split('.')[-1]
    if type=='pdf':
        loader=PyMuPDFLoader(file_path)
        loaders.append(loader)
    if type=='md':
        loader=UnstructuredMarkdownLoader(file_path)
        loaders.append(loader)
    
texts=[]
for loader in loaders:
    texts.extend(loader.load())

# 分割文档
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
split_docs=text_splitter.split_documents(texts)

#构建chroma向量库
from langchain.embeddings.openai import OpenAIEmbeddings

embedding=OpenAIEmbeddings()
persist_directory='data_base/vector_db/chroma_1'

from langchain.vectorstores.chroma import Chroma
vectordb=Chroma.from_documents(documents=split_docs[:20],embedding=embedding,persist_directory=persist_directory)
vectordb.persist()

question='大模型有什么用'
sim_docs=vectordb.similarity_search(query=question,k=4)
print(sim_docs[1].page_content)

