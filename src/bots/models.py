from langchain_core.messages import SystemMessage, HumanMessage
from src.configs.config import settings

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from src.db.vector_store import chroma_db


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


ANSWER_GENERATION_PROMPT = """You are a QA system answering questions using ONLY the context provided below.

The context contains:
1. SUPPORTING DOCUMENT EXCERPTS: raw text from source articles.

Instructions:
- First, check the DOCUMENT EXCERPTS for specific details that answer the question.
- For Yes/No questions: look for SPECIFIC evidence in the excerpts that either 
  supports or contradicts the claim. Do NOT guess.
  If the excerpts discuss the topic, you CAN determine Yes or No from their content.
- Only respond "Unknown" if the context truly contains NO relevant information 
  about the question topic. If you see relevant excerpts, give an answer.
- Be concise: answer with a single entity name, date, short phrase, or Yes or No.
- Do NOT explain your reasoning.

Context:
{context}

Question: {question}"""

llm = ChatOpenAI(model=settings.CHAT_MODEL, temperature=0, api_key=settings.OPENAI_API_KEY)
def ask_vector_rag_direct(query: str, chroma_db = chroma_db, llm = llm) -> str:
    docs = chroma_db.similarity_search(query, k=3)
    
    context_parts = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get('title', f'Source {i+1}')
        context_parts.append(f"[Source: {title}]\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    response = llm.invoke([HumanMessage(content=ANSWER_GENERATION_PROMPT.format(context=context, question=query))])
    
    return response.content.strip()