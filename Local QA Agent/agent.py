import os
from langchain.document_loaders import TextLoader, CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Defina sua chave da OpenAI. Melhor prática: use variáveis de ambiente.
os.environ["OPENAI_API_KEY"] = "sua-chave-aqui"

# Carregar os arquivos locais (pode ser .txt, .md ou .csv)
loaders = [
    TextLoader("data/processos.md"), 
    TextLoader("data/colaboradores.txt"),  
    CSVLoader(file_path="data/ferramentas.csv")  
]

documents = []
# Carregar o conteúdo de todos os arquivos
for loader in loaders:
    documents.extend(loader.load())

embeddings = OpenAIEmbeddings()

# Criar um índice vetorial usando FAISS para busca eficiente
vectorstore = FAISS.from_documents(documents, embeddings)

# Criar o agente de QA
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),  # Configurar o modelo de linguagem
    retriever=vectorstore.as_retriever()  # Usar o vetor de documentos como fonte de consulta
)

#Interação
print("Pergunte sobre os processos internos. (digite 'sair' para encerrar)")
while True:
    query = input("Você: ")  # Solicitar a pergunta do usuário
    if query.lower() == "sair":  # Permitir sair do loop
        break
    resposta = qa.run(query)  # Obter a resposta do agente
    print(f"Agente: {resposta}")  # Exibir a resposta

