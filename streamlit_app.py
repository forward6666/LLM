import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv,find_dotenv
import os
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import tempfile
from openai import OpenAI


load_dotenv(find_dotenv())

def generate_response(input_text,openai_api_key):
    llm=ChatOpenAI(temperature=0.7,openai_api_key=os.environ['OPENAI_API_KEY'],streaming=True)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    #st.info(output)
    return output

def embedding_base(content):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    split_docs=text_splitter.split_documents(content)
    embedding=OpenAIEmbeddings()
    vectordb=Chroma.from_documents(documents=split_docs,embedding=embedding)
    return vectordb

def get_init_vectordb():
    embedding=OpenAIEmbeddings()
    persist_directory='data_base/vector_db\chroma_1'
    vectordb=Chroma(persist_directory=persist_directory,embedding_function=embedding)
    return vectordb

def get_vectordb(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())

    type=path.split('.')[-1]
    if type=='pdf':
        loader = PyMuPDFLoader(path)
        documents = loader.load()
    elif type=='md':
        loader=UnstructuredMarkdownLoader(path)
        documents=loader.load()

    vectordb=embedding_base(documents)
    return vectordb

def get_chat_qa_chain(question,openai_api_key,vectordb):

    llm=ChatOpenAI(temperature=0,openai_api_key=os.environ['OPENAI_API_KEY'],streaming=True)
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    retriever=vectordb.as_retriever()
    qa=ConversationalRetrievalChain.from_llm(llm,retriever=retriever,memory=memory)
    result=qa({'question':question})
    return result['answer']

def get_qa_chain(question:str,openai_api_key:str,vectordb):

    retriever=vectordb.as_retriever()
    llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0,openai_api_key = os.environ['OPENAI_API_KEY'],streaming=True)
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    qa_chain_prompt=PromptTemplate(input_variables=['context','question'],template=template)
    qa_chain=RetrievalQA.from_chain_type(llm,retriever=retriever,
                                         return_source_documents=True,
                                         chain_type_kwargs={'prompt':qa_chain_prompt}
                                         )
    result=qa_chain({'query':question})
    return result['result']
    
def unique_response(prompt):
    client=OpenAI()
    messages=[{'role':'system','content':'你是一个清朝的太监，而我是一位清朝的皇帝，你需要一直用太监的语气回答我的问题，回答问题时需要称呼我为皇上，而你需要自称为奴才，特别注意，每次回答问题前都需要先回复：嗻，回皇上'},
              {'role':'user','content':prompt}]
    response=client.chat.completions.create(model='gpt-3.5-turbo',temperature=0.7,messages=messages)
    if len(response.choices)>0:
        return response.choices[0].message.content
    else:
        st.error('错误',icon='⚡')


def main():
    st.title('皇帝模拟器')
    st.header("👸👑👑👑**吾皇**万岁万岁万万岁")
    st.divider()
    st.caption("*全心全意为皇上服务！*")
    with st.sidebar:
        # openai_api_key = st.text_input('OpenAI API Key', type='password')
        selected_method = st.selectbox("选择模式", ["None","qa_chain", "chat_qa_chain",'服务皇帝模式'])
        uploaded_file=st.file_uploader('上交奏折',type=['pdf','md'])
        c_vectordb_button=st.button('批阅奏折')
        if c_vectordb_button:
            if uploaded_file is not None:
                vectordb=get_vectordb(uploaded_file)
                st.success('奏折处理完毕',icon='✨')
            else:
                st.toast('未上交奏折',icon="😥")
                vectordb=get_init_vectordb()
            
 

    # 用于跟踪对话历史
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    messages = st.container(height=450)
    messages.chat_message("assistant").write('皇上请吩咐！')
    if prompt := st.chat_input("Say something"):
            # 将用户输入添加到对话历史中
        st.session_state.chat_history.append({"role": "user", "text": prompt})

        if selected_method == "None":
                # 调用 respond 函数获取回答
            answer = generate_response(prompt)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt,vectordb)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(prompt,vectordb)
        elif selected_method =='服务皇帝模式':
            answer =unique_response(prompt)
            # 调用 respond 函数获取回答

            # 检查回答是否为 None
        if answer is not None:
        #         # 将LLM的回答添加到对话历史中
            st.session_state.chat_history.append({"role": "assistant", "text": answer})

            # 显示整个对话历史
        for history in st.session_state.chat_history:
            if history["role"] == "user":
                messages.chat_message("user").write(history["text"])
            elif history["role"] == "assistant":
                messages.chat_message("assistant").write(history["text"])
            
if __name__=='__main__':
    main()
        