from typing import Union
from dotenv import load_dotenv
from fastapi import FastAPI
import os
from src.core.assistant import Assistant
from src.core.db import DB

chatbot = Assistant()
app = FastAPI()


@app.get("/ask")
def ask(question: str):
    return chatbot.ask(question)


@app.get("/load_documents")
def load_documents():
    db = DB()
    db._load_documents()
