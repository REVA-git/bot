from typing import Union
from dotenv import load_dotenv
from fastapi import FastAPI
import os
from src.core.assistant import Assistant

load_dotenv()

# print(os.getenv("GOOGLE_API_KEY"))

chatbot = Assistant()
app = FastAPI()


@app.get("/ask")
def ask(question: str):
    return chatbot.ask(question)


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
