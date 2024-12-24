import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph


def get_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0)
    return llm


class Assistant:
    def __init__(self):
        self.llm = get_llm()
        self.graph = self._create_graph(self._assistant)

    def ask(self, question: str):
        config = {"configurable": {"thread_id": 1}, "recursion_limit": 40}
        messages = [HumanMessage(content=question)]
        return self.graph.stream({"messages": messages}, config, stream_mode="updates")

    def _assistant(self, state: MessagesState):
        sys_msg = self._load_prompt()
        return {"messages": [self.llm.invoke([sys_msg] + state["messages"])]}

    def _load_prompt(self):
        with open("assets/system_prompt.md", "r") as file:
            prompt = file.read()
        return SystemMessage(content=prompt)

    def _create_graph(self, assistant_node):
        builder = StateGraph(MessagesState)
        builder.add_node("assistant", assistant_node)
        builder.add_edge(START, "assistant")
        return builder.compile()
