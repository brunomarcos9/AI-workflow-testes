import os
from langchain.document_loaders import TextLoader, CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings  # Importação corrigida
from langchain.chat_models import ChatOpenAI  # Importação corrigida
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Verificar se a chave foi carregada corretamente
if api_key is None:
    print("A chave da OpenAI não foi encontrada. Verifique o arquivo .env.")
else:
    os.environ["OPENAI_API_KEY"] = api_key

# Carregar os arquivos locais
loaders = [
    TextLoader("data/processos.md"), 
    TextLoader("data/colaboradores.txt"),  
    CSVLoader(file_path="data/ferramentas.csv")  
]

documents = []
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

# Interação
print("Pergunte sobre os processos internos. (digite 'sair' para encerrar)")
while True:
    query = input("Você: ")  # Solicitar a pergunta do usuário
    if query.lower() == "sair":  # Permitir sair do loop
        break
    resposta = qa.run(query)  # Obter a resposta do agente
    print(f"Agente: {resposta}")  # Exibir a resposta
