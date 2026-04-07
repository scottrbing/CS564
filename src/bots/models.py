from langchain_core.messages import SystemMessage, HumanMessage
from src.configs.config import settings

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class RagModel:
    _instance = None

    def _build_agent(self, rag_tools):
        llm = ChatOpenAI(model=settings.CHAT_MODEL, temperature=0, api_key=settings.OPENAI_API_KEY)

        @dynamic_prompt
        def system_prompt(request: ModelRequest):
            return f"""You are a QA agent being evaluated on a database benchmark. 
                You will be asked complex, multi-hop questions. You have access to tools to retrieve context.
                You must use your tools to find the information needed to answer the question.

                CRITICAL FORMATTING INSTRUCTIONS:
                1. Your final output must be extremely concise. 
                2. Use only a single word, entity name, date, or short phrase.
                3. Do NOT write full sentences. 
                4. Do NOT explain your reasoning.
                5. Do NOT use introductory filler (e.g., never write "The answer is...").

                Answer the user's question directly."""


        return create_agent(
            model=llm,
            tools= rag_tools,
            middleware=[system_prompt]
        )

    @classmethod
    def init(cls):
        if cls._instance is None:
            cls._instance = cls()
        if not hasattr(cls._instance, "vector_agent"):
            # Replace None with our RAG tools
            cls._instance.vector_agent = cls._instance._build_agent(None)
        if not hasattr(cls._instance, "graph_agent"):
            # Replace None with our RAG tools
            cls._instance.graph_agent = cls._instance._build_agent(None)
        return cls._instance
    
    def ask(self, agent, user_query: str):
        runtime_context = {}
        # We ONLY pass the HumanMessage. 
        print(f'You: {user_query}')
        response_state = agent.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            context=runtime_context
        )
        
        return response_state["messages"][-1].content

    def ask_vector_rag(self, user_query: str):
        return  self.ask(self.vector_agent, user_query)
    
    def ask_graph_rag(self, user_query: str):
        return  self.ask(self.graph_agent, user_query)
