from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
import os

DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH")


class DB:
    def __init__(self):
        self.embedding = FastEmbedEmbeddings()
        self.store = Chroma(
            persist_directory="./chroma_db", embedding_function=self.embedding
        )
        self._load_documents()

    def get_retriever(self):
        return self.store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.3,
            },
        )

    def add_document(self, file_path: str):
        chunks = self._load_document(file_path)
        self.store.add_documents(chunks)

    def _load_documents(self):
        for file in os.listdir(DOCUMENTS_PATH):
            if (
                file.endswith(".pdf")
                or file.endswith(".md")
                or file.endswith(".docx")
                or file.endswith(".txt")
            ):
                self.add_document(os.path.join(DOCUMENTS_PATH, file))

    @staticmethod
    def _load_document(file_path: str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        # elif file_path.endswith(".md"):
        #     loader = MarkdownTextSplitter(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".txt") or file_path.endswith(".md"):
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        chunks = loader.load_and_split(text_splitter)
        print(f"Split documents into {len(chunks)} chunks.")
        return chunks


vector_db = DB()
