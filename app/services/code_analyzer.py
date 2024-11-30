import os
from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

class CodeAnalyzer:
    def __init__(self, repo_url: str, openai_api_key: str):
        self.repo_url = repo_url
        self.openai_api_key = openai_api_key
        self.repo_path = "temp_repo"
        self.qa = None
        
    async def initialize(self):
        # Clone repository
        if os.path.exists(self.repo_path):
            os.system(f"rm -rf {self.repo_path}")
        os.makedirs(self.repo_path)
        Repo.clone_from(self.repo_url, self.repo_path)
        
        # Load and process documents
        loader = GenericLoader.from_filesystem(
            self.repo_path + '/',
            glob="**/*",
            suffixes=[".js", ".jsx", ".tsx", "ts", ".py"],
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
        )
        
        documents = loader.load()
        
        # Split documents
        documents_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=2000,
            chunk_overlap=200
        )
        
        texts = documents_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            disallowed_special=()
        )
        
        vectordb = Chroma.from_documents(
            texts,
            embedding=embeddings,
            persist_directory='./data'
        )
        
        # Initialize LLM and QA chain
        llm = ChatOpenAI(
            model_name="gpt-4",
            openai_api_key=self.openai_api_key
        )
        
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True
        )
        
        self.qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8}
            ),
            memory=memory
        )
    
    async def analyze_code(self, question: str):
        if not self.qa:
            raise Exception("Analyzer not initialized. Call initialize() first.")
        
        result = await self.qa.ainvoke({"question": question})
        return result["answer"] 