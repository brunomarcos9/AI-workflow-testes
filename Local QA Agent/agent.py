import os
from langchain.document_loaders import TextLoader, CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI 
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

#Verifica api OPENAI
if not api_key:
    raise ValueError("A chave da OpenAI não foi encontrada. Verifique o arquivo .env.")

#Arquivos base para API usar
loaders = [
    TextLoader("data/processos.md"), 
    TextLoader("data/colaboradores.txt"),  
    CSVLoader(file_path="data/ferramentas.csv")  
]

documents = []
for loader in loaders:
    documents.extend(loader.load())

embeddings = OpenAIEmbeddings()

#Vetor
vectorstore = FAISS.from_documents(documents, embeddings)

#Agente de QA
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),  
    retriever=vectorstore.as_retriever()  #Consulta ao vetor
)

# Interação
print("Pergunte sobre os processos internos. (digite 'sair' para encerrar)")
while True:
    query = input("Você: ")  
    if query.lower() == "sair":  
        break
    resposta = qa.run(query)  
    print(f"Agente: {resposta}")
