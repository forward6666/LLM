from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader


loader = PyMuPDFLoader('data_base\knowledge_db\pumkin_book\pumpkin_book.pdf')
pdf_pages = loader.load()
pdf_page = pdf_pages[1]
# print(f"文档内容{pdf_page.page_content}",
#       f"文档描述性语言{pdf_page.metadata}",
#       sep="\n-----------\n")


import re
pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
# print(pdf_page.page_content)
pdf_page.page_content = pdf_page.page_content.replace('•', '')
pdf_page.page_content = pdf_page.page_content.replace(' ', '')
# print(pdf_page.page_content)


from langchain.text_splitter import RecursiveCharacterTextSplitter
chunk_size = 500
chunk_overlap = 50
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
text_splitter.split_text(pdf_page.page_content[:1000])
split_docs = text_splitter.split_documents(pdf_pages)
print(len(split_docs))