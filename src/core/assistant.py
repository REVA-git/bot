from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState

from src.core.db import DB
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def get_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0)
    return llm


class Assistant:
    def __init__(self):
        self.db = DB()
        self.graph = self._create_graph()

    def ask(self, question: str):
        config = {"configurable": {"thread_id": 1}, "recursion_limit": 40}
        messages = [HumanMessage(content=question)]
        return self.graph.invoke({"messages": messages}, config, stream_mode="updates")

    def _assistant(self, state):
        question = state["messages"][-1].content
        llm = get_llm()
        with open("assets/system_prompt.md", "r") as file:
            prompt = file.read()

        rag_prompt_template = ChatPromptTemplate.from_template(prompt)
        retriever = self.db.get_retriever()

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt_template
            | llm
            | StrOutputParser()
        )
        documents = retriever.invoke(question)
        answer = rag_chain.invoke({"question": question, "context": documents})
        return {"messages": [AIMessage(content=answer)]}

    def _create_graph(self):
        builder = StateGraph(MessagesState)
        builder.add_node("assistant", self._assistant)
        builder.set_entry_point("assistant")
        builder.set_finish_point("assistant")
        return builder.compile()
