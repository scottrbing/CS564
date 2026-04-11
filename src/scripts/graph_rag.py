import os
from flask.cli import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI, api_key


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("Loaded key:", client.api_key[:10] if client.api_key else "NOT FOUND")


# ----------------- CONFIGURATION -----------------
# client = OpenAI(api_key="your-openai-key")
#client = OpenAI(api_key="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")

URI = "bolt://localhost:7687"
AUTH = ("CS564FinalProject", "!I10wrk01") 
driver = GraphDatabase.driver(URI, auth=AUTH)

# MUST match your database name exactly
DB_NAME = "CS564 Final Project"

def query_graph(tx, keyword):
    # Improved query: Checks for the word itself AND similar words
    query = """
    MATCH (w:Word {text: $kw})
    OPTIONAL MATCH (w)-[:SIMILAR_TO]-(w2:Word)
    WITH collect(DISTINCT w) + collect(DISTINCT w2) AS targetWords
    UNWIND targetWords AS target
    MATCH (d:Document)-[:CONTAINS]->(target)
    RETURN d.id AS doc, collect(DISTINCT target.text) AS terms
    LIMIT 5
    """
    return list(tx.run(query, kw=keyword))

def extract_keyword(question):
    # Simple extraction: takes the longest word to avoid "the", "is", etc.
    words = [w.strip("?!.,").lower() for w in question.split()]
    stop = {"the","is","a","an","what","about","tell","me","of","explain","show"}
    keywords = [w for w in words if w not in stop and len(w) > 2]
    return keywords[0] if keywords else words[0]

def build_context(keyword):
    try:
        # Explicitly targeting your project database
        with driver.session(database=DB_NAME) as session:
            results = session.execute_read(query_graph, keyword)

        if not results:
            return f"No graph data found for keyword: '{keyword}'"

        context = "Relevant Data from Knowledge Graph:\n"
        for r in results:
            context += f"- Document {r['doc']} contains terms: {', '.join(r['terms'])}\n"
        return context
        
    except Exception as e:
        return f"Database error: {str(e)}"

def ask_llm(question):
    keyword = extract_keyword(question)
    context = build_context(keyword)

    prompt = f"""
    You are an AI assistant powered by a Neo4j Graph.
    
    CONTEXT FROM GRAPH:
    {context}
    
    USER QUESTION:
    {question}
    
    INSTRUCTION: Use the context above to answer. If the context is missing, 
    answer based on your general knowledge but mention the graph was empty.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

def main():
    print(f"--- Connected to {DB_NAME} ---")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        answer = ask_llm(user_input)
        print(f"\nAI: {answer}\n")

    driver.close()

if __name__ == "__main__":
    main()