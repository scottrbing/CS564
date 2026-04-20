"""
Hybrid Graph+Vector RAG Retrieval Pipeline
============================================
Uses the knowledge graph for what it's good at (entity discovery and 
relationship traversal) and vector search for what it's good at 
(finding relevant document chunks with detailed content).

Pipeline:
  1. LLM extracts key entities from the user's question
  2. Fuzzy-match those entities to actual Neo4j node IDs
  3. Traverse the graph 2 hops to discover connected entities and relationships
  4. Build an enriched search query from discovered entities/relationships
  5. Vector search with BOTH the original query AND the enriched query
  6. Pass graph triples (structured reasoning) + vector chunks (detailed content)
     to the LLM for answer generation

Why this works for multi-hop:
  - Pure vector RAG: searches with the original question, may miss documents 
    about entities that aren't mentioned in the question but are part of the 
    reasoning chain.
  - Hybrid: the graph discovers intermediate entities (the "hops"), then 
    vector search finds documents about those entities too. The graph tells
    you WHERE to look; vector search finds WHAT those documents say.

Usage:
  retriever = HybridGraphVectorRetriever(neo4j_graph, chroma_db, openai_api_key)
  answer = retriever.answer("What company acquired Instagram?")
"""

import json
import re
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ---------------------------------------------------------------------------
# STEP 1: Entity Extraction from the Query (unchanged)
# ---------------------------------------------------------------------------

ENTITY_EXTRACTION_PROMPT = """You are an entity extractor for a knowledge graph search system.

Given a user question, extract the key NAMED ENTITIES (proper nouns) that should be searched for in the graph.

Rules:
- Extract ONLY proper nouns: specific people, companies, products, publications, places, events, regulations.
- NEVER extract common English words, even if they appear important in the question.
  REJECT: "team", "hype", "group", "bettors", "celebrity", "company", "CEO", 
  "revenue", "date", "city", "coverage", "plans", "report", "article", "information"
- Ask yourself: "Would this word be capitalized in a news headline as a proper noun?" 
  If no, do NOT extract it.
- For each entity, provide plausible ALIASES — but only meaningful multi-word 
  variations or well-known abbreviations, NOT single common words.
  GOOD aliases: "Sam Bankman-Fried" -> ["SBF", "Bankman-Fried"]
  BAD aliases: "Layton Williams" -> ["Williams"] (too generic, will match wrong nodes)
  BAD aliases: "Billy Elliot" -> ["Billy"] (too generic)
  BAD aliases: "The Independent" -> ["Independent"] (too generic)
- Return 1-4 entities maximum. Focus on the most specific, identifiable ones.
- Think about what the knowledge graph might store the entity name as. 
  For example, "OpenAI" might be stored as "Openai" (title case).

Respond ONLY with a JSON array, no markdown fences, no explanation:
[
  {{"name": "Primary Name", "aliases": ["alias1", "alias2"]}},
  ...
]

Question: {question}"""


def extract_entities(llm: ChatOpenAI, question: str) -> list[dict]:
    """Use the LLM to extract named entities and aliases from the question."""
    response = llm.invoke([
        HumanMessage(content=ENTITY_EXTRACTION_PROMPT.format(question=question))
    ])
    
    text = response.content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    
    try:
        entities = json.loads(text)
        for entity in entities:
            entity["aliases"] = [
                a for a in entity.get("aliases", [])
                if (len(a) >= 4 and " " in a) or len(a) >= 5
            ]
        return entities
    except json.JSONDecodeError:
        print(f"[WARN] Failed to parse entity extraction output: {text}")
        return []


# ---------------------------------------------------------------------------
# STEP 2: Fuzzy Entity Matching in Neo4j (unchanged)
# ---------------------------------------------------------------------------

FUZZY_MATCH_CYPHER = """
UNWIND $candidates AS candidate
MATCH (n)
WHERE n.id IS NOT NULL
  AND toLower(n.id) CONTAINS toLower(candidate)
WITH n, candidate,
     CASE 
       WHEN toLower(n.id) = toLower(candidate) THEN 0
       WHEN n.id STARTS WITH candidate THEN 1
       ELSE 2 
     END AS match_tier,
     toFloat(size(candidate)) / toFloat(size(n.id)) AS name_coverage
WITH n, candidate, match_tier, name_coverage
WHERE name_coverage >= 0.3 OR match_tier = 0
ORDER BY match_tier ASC, name_coverage DESC
RETURN DISTINCT n.id AS node_id, labels(n) AS node_labels, 
       candidate AS matched_on, match_tier, name_coverage
LIMIT $max_matches
"""


def match_entities_in_graph(graph, entities: list[dict], max_per_entity: int = 2) -> list[dict]:
    all_matches = []
    seen_ids = set()
    
    for entity in entities:
        candidates = [entity["name"]] + entity.get("aliases", [])
        
        try:
            results = graph.query(
                FUZZY_MATCH_CYPHER,
                params={"candidates": candidates, "max_matches": max_per_entity * 2}
            )
            
            entity_matches = 0
            for row in results:
                if entity_matches >= max_per_entity:
                    break
                nid = row["node_id"]
                if nid not in seen_ids:
                    seen_ids.add(nid)
                    all_matches.append({
                        "node_id": nid,
                        "labels": row["node_labels"],
                        "matched_on": row["matched_on"],
                        "match_tier": row["match_tier"],
                        "name_coverage": row["name_coverage"],
                        "original_entity": entity["name"]
                    })
                    entity_matches += 1
        except Exception as e:
            print(f"[WARN] Fuzzy match failed for {entity['name']}: {e}")
    
    return all_matches


# ---------------------------------------------------------------------------
# STEP 3: Graph Traversal (simplified — focused on entity discovery)
# ---------------------------------------------------------------------------

# We don't need source texts from the graph anymore — vector search handles that.
# The graph's job is ONLY to discover entities and their relationships.

SUBGRAPH_TRAVERSAL_CYPHER = """
MATCH (anchor)
WHERE anchor.id = $node_id

OPTIONAL MATCH (anchor)-[r1]-(n1)
WHERE NOT n1:Document AND NOT n1:Date

OPTIONAL MATCH (n1)-[r2]-(n2)
WHERE NOT n2:Document AND NOT n2:Date
  AND n2.id <> anchor.id

WITH anchor, 
     COLLECT(DISTINCT {
       src: anchor.id, 
       rel: type(r1), 
       tgt: n1.id,
       hop: 1
     }) AS hop1_triples,
     COLLECT(DISTINCT {
       src: n1.id, 
       rel: type(r2), 
       tgt: n2.id,
       hop: 2
     }) AS hop2_triples

WITH hop1_triples + hop2_triples AS all_triples
UNWIND all_triples AS t
WITH DISTINCT t
WHERE t.src IS NOT NULL AND t.tgt IS NOT NULL AND t.rel IS NOT NULL
RETURN t.src AS source, t.rel AS relationship, t.tgt AS target, t.hop AS hop
LIMIT 150
"""


def traverse_subgraph(graph, matched_nodes: list[dict]) -> list[dict]:
    """Traverse the graph and return raw triples."""
    all_triples = []
    seen_triples = set()
    
    for node in matched_nodes:
        nid = node["node_id"]
        try:
            results = graph.query(
                SUBGRAPH_TRAVERSAL_CYPHER,
                params={"node_id": nid}
            )
            for row in results:
                triple_key = (row["source"], row["relationship"], row["target"])
                if triple_key not in seen_triples:
                    seen_triples.add(triple_key)
                    all_triples.append({
                        "source": row["source"],
                        "relationship": row["relationship"],
                        "target": row["target"],
                        "hop": row["hop"],
                        "anchor": nid
                    })
        except Exception as e:
            print(f"[WARN] Traversal failed for {nid}: {e}")
    
    return all_triples


# ---------------------------------------------------------------------------
# STEP 4: Filter Graph Triples (same logic as before)
# ---------------------------------------------------------------------------

def filter_triples_by_relevance(triples: list[dict], matched_nodes: list[dict]) -> list[dict]:
    anchor_ids = set(n["node_id"] for n in matched_nodes)
    
    anchor_to_original = {}
    for node in matched_nodes:
        anchor_to_original[node["node_id"]] = node.get("original_entity", node["node_id"])
    
    distinct_originals = set(anchor_to_original.values())
    is_single_entity = len(distinct_originals) <= 1
    
    anchor_neighborhoods = {}
    for node in matched_nodes:
        nid = node["node_id"]
        anchor_neighborhoods[nid] = {nid}
        for t in triples:
            if t["anchor"] == nid:
                anchor_neighborhoods[nid].add(t["source"])
                anchor_neighborhoods[nid].add(t["target"])
    
    key_triples = []
    bridging_triples = []
    direct_triples = []
    seen_key_pairs = set()
    
    for t in triples:
        src, tgt = t["source"], t["target"]
        anchor = t["anchor"]
        
        if src in anchor_ids and tgt in anchor_ids:
            src_orig = anchor_to_original.get(src, src)
            tgt_orig = anchor_to_original.get(tgt, tgt)
            if src_orig != tgt_orig:
                pair = tuple(sorted([src_orig, tgt_orig]))
                dedup_key = (pair, t["relationship"])
                if dedup_key not in seen_key_pairs:
                    seen_key_pairs.add(dedup_key)
                    key_triples.append(t)
                continue
        
        is_bridging = False
        if not is_single_entity:
            for other_anchor, neighborhood in anchor_neighborhoods.items():
                if anchor_to_original.get(other_anchor) != anchor_to_original.get(anchor):
                    if src in neighborhood or tgt in neighborhood:
                        is_bridging = True
                        break
        
        if is_bridging:
            bridging_triples.append(t)
        elif t["hop"] == 1:
            direct_triples.append(t)
    
    if is_single_entity:
        filtered = direct_triples[:30]
    else:
        MAX_TOTAL = 50
        filtered = key_triples[:8]
        remaining = MAX_TOTAL - len(filtered)
        filtered.extend(bridging_triples[:min(25, remaining)])
        remaining = MAX_TOTAL - len(filtered)
        filtered.extend(direct_triples[:min(15, remaining)])
    
    return filtered


# ---------------------------------------------------------------------------
# STEP 5: Build Enriched Vector Query from Graph Discoveries
# ---------------------------------------------------------------------------

def build_enriched_queries(question: str, entities: list[dict], 
                           matched_nodes: list[dict], filtered_triples: list[dict]) -> list[str]:
    """
    Build multiple search queries using graph-discovered entities.
    
    The key insight: the graph tells us about entities the user DIDN'T mention
    but that are part of the reasoning chain. We search for those too.
    
    Returns a list of queries to run against the vector store.
    """
    queries = []
    
    # Query 1: Always include the original question
    queries.append(question)
    
    # Collect entities discovered via the graph that weren't in the original question
    original_entity_names = set()
    for e in entities:
        original_entity_names.add(e["name"].lower())
        for a in e.get("aliases", []):
            original_entity_names.add(a.lower())
    
    # Find discovered entities from KEY and bridging triples
    discovered_entities = set()
    anchor_ids = set(n["node_id"] for n in matched_nodes)
    
    for t in filtered_triples:
        for endpoint in [t["source"], t["target"]]:
            if endpoint and endpoint.lower() not in original_entity_names:
                # This entity was discovered through graph traversal
                discovered_entities.add(endpoint)
    
    # Query 2: Combine key discovered entities with the original question topic
    if discovered_entities:
        # Pick the most likely relevant discovered entities (from KEY triples first)
        key_discovered = set()
        for t in filtered_triples:
            if t.get("source") in anchor_ids or t.get("target") in anchor_ids:
                for endpoint in [t["source"], t["target"]]:
                    if endpoint and endpoint not in anchor_ids:
                        key_discovered.add(endpoint)
        
        if key_discovered:
            # Build a query combining original entities with discovered ones
            original_names = [e["name"] for e in entities[:3]]
            discovered_names = list(key_discovered)[:3]
            enriched = " ".join(original_names + discovered_names)
            queries.append(enriched)
    
    # Query 3: For each pair of original entities, search them together
    entity_names = [e["name"] for e in entities]
    if len(entity_names) >= 2:
        for i in range(min(len(entity_names), 3)):
            for j in range(i + 1, min(len(entity_names), 3)):
                pair_query = f"{entity_names[i]} {entity_names[j]}"
                queries.append(pair_query)
    
    return queries[:5]  # Cap at 5 queries


# ---------------------------------------------------------------------------
# STEP 6: Vector Search with Enriched Queries
# ---------------------------------------------------------------------------

def vector_search_with_enrichment(chroma_db, queries: list[str], k_per_query: int = 3) -> list[str]:
    """
    Run multiple vector searches and deduplicate results.
    Returns unique document chunks, prioritizing those that appear in multiple queries.
    """
    chunk_scores = {}  # text -> count of queries it appeared in
    chunk_order = []   # preserve first-seen order
    
    for query in queries:
        try:
            docs = chroma_db.similarity_search(query, k=k_per_query)
            for doc in docs:
                text = doc.page_content
                if text not in chunk_scores:
                    chunk_scores[text] = {
                        "count": 0,
                        "title": doc.metadata.get("title", "Unknown"),
                        "text": text
                    }
                    chunk_order.append(text)
                chunk_scores[text]["count"] += 1
        except Exception as e:
            print(f"[WARN] Vector search failed for query: {e}")
    
    # Sort by number of query hits (docs appearing in multiple searches are more relevant)
    sorted_chunks = sorted(chunk_order, key=lambda t: -chunk_scores[t]["count"])
    
    results = []
    for text in sorted_chunks[:8]:  # Top 8 chunks
        info = chunk_scores[text]
        results.append({
            "title": info["title"],
            "text": info["text"],
            "relevance_hits": info["count"]
        })
    
    return results


# ---------------------------------------------------------------------------
# STEP 7: Format Combined Context
# ---------------------------------------------------------------------------

def format_hybrid_context(filtered_triples: list[dict], vector_chunks: list[dict],
                          matched_nodes: list[dict]) -> str:
    """Combine graph triples and vector-retrieved chunks into a single context."""
    parts = []
    anchor_ids = set(n["node_id"] for n in matched_nodes)
    
    # Section 1: Graph-discovered relationships (compact, structured)
    if filtered_triples:
        parts.append("=== KNOWLEDGE GRAPH RELATIONSHIPS ===")
        for t in filtered_triples:
            src, rel, tgt = t["source"], t["relationship"], t["target"]
            if src in anchor_ids and tgt in anchor_ids:
                parts.append(f"[KEY] ({src}) -[{rel}]-> ({tgt})")
            else:
                parts.append(f"({src}) -[{rel}]-> ({tgt})")
    
    # Section 2: Vector-retrieved document chunks (detailed content)
    if vector_chunks:
        parts.append("\n=== RETRIEVED DOCUMENT EXCERPTS ===")
        for i, chunk in enumerate(vector_chunks, 1):
            title = chunk["title"]
            text = chunk["text"][:800] + "..." if len(chunk["text"]) > 800 else chunk["text"]
            parts.append(f"\n[Source: {title}]\n{text}")
    
    if not parts:
        return "No relevant information found."
    
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# STEP 8: Answer Generation
# ---------------------------------------------------------------------------

ANSWER_GENERATION_PROMPT = """You are a QA system answering questions using ONLY the context provided below.
The context contains two types of information:
1. KNOWLEDGE GRAPH RELATIONSHIPS: structured triples showing how entities are connected.
   Triples marked [KEY] directly connect entities from the question — pay special attention to these.
2. RETRIEVED DOCUMENT EXCERPTS: relevant text passages from source articles.

Instructions:
- Use the graph relationships to understand HOW entities are connected (the reasoning chain).
- Use the document excerpts for specific details, dates, quotes, and nuanced answers.
- Both sources together should give you enough to answer. If not, respond with "Unknown".
- Be concise: answer with a single entity name, date, short phrase, or Yes/No.
- Do NOT explain your reasoning or say "based on the context".
- For Yes/No questions, carefully check whether the excerpts SUPPORT or CONTRADICT the claim.

Context:
{context}

Question: {question}"""


def generate_answer(llm: ChatOpenAI, question: str, context: str) -> str:
    response = llm.invoke([
        HumanMessage(content=ANSWER_GENERATION_PROMPT.format(
            context=context,
            question=question
        ))
    ])
    return response.content.strip()


# ---------------------------------------------------------------------------
# MAIN HYBRID RETRIEVER CLASS
# ---------------------------------------------------------------------------

class HybridGraphVectorRetriever:
    """
    Hybrid Graph+Vector RAG pipeline:
      query -> entity extraction -> graph match -> graph traversal 
      -> enriched vector search -> combined context -> answer
    
    The graph finds the RIGHT entities and connections.
    Vector search finds the RIGHT document content.
    Together they handle multi-hop reasoning with detailed answers.
    """
    
    def __init__(self, neo4j_graph, chroma_db, openai_api_key: str,
                 extraction_model: str = "gpt-4o-mini",
                 answer_model: str = "gpt-4o-mini"):
        """
        Args:
            neo4j_graph: LangChain Neo4jGraph instance
            chroma_db: Chroma vector store instance (with .similarity_search method)
            openai_api_key: OpenAI API key
        """
        self.graph = neo4j_graph
        self.chroma_db = chroma_db
        self.extraction_llm = ChatOpenAI(
            temperature=0, model=extraction_model, api_key=openai_api_key
        )
        self.answer_llm = ChatOpenAI(
            temperature=0, model=answer_model, api_key=openai_api_key
        )
    
    def retrieve_context(self, question: str, verbose: bool = False) -> str:
        """Full retrieval pipeline: graph discovery + vector search."""
        
        # Step 1: Extract entities
        entities = extract_entities(self.extraction_llm, question)
        if verbose:
            print(f"[Step 1] Extracted entities: {json.dumps(entities, indent=2)}")
        
        if not entities:
            # Fallback: pure vector search with original question
            if verbose:
                print("[Fallback] No entities extracted, using pure vector search")
            chunks = vector_search_with_enrichment(self.chroma_db, [question], k_per_query=5)
            return format_hybrid_context([], chunks, [])
        
        # Step 2: Fuzzy match to graph
        matched_nodes = match_entities_in_graph(self.graph, entities, max_per_entity=2)
        if verbose:
            for m in matched_nodes:
                print(f"  [Step 2] Matched: '{m['node_id']}' "
                      f"(via '{m['matched_on']}', tier={m['match_tier']}, "
                      f"coverage={m['name_coverage']:.2f})")
        
        # Step 3: Traverse graph
        if matched_nodes:
            raw_triples = traverse_subgraph(self.graph, matched_nodes)
            filtered_triples = filter_triples_by_relevance(raw_triples, matched_nodes)
            if verbose:
                print(f"[Step 3] Graph: {len(filtered_triples)} triples "
                      f"(from {len(raw_triples)} raw)")
        else:
            filtered_triples = []
            if verbose:
                print("[Step 3] No graph matches, skipping traversal")
        
        # Step 4: Build enriched vector queries
        enriched_queries = build_enriched_queries(
            question, entities, matched_nodes, filtered_triples
        )
        if verbose:
            print(f"[Step 4] Vector queries: {enriched_queries}")
        
        # Step 5: Vector search
        vector_chunks = vector_search_with_enrichment(
            self.chroma_db, enriched_queries, k_per_query=3
        )
        if verbose:
            print(f"[Step 5] Retrieved {len(vector_chunks)} unique chunks")
            for c in vector_chunks:
                print(f"  - [{c['relevance_hits']} hits] {c['title'][:60]}")
        
        # Step 6: Format combined context
        context = format_hybrid_context(filtered_triples, vector_chunks, matched_nodes)
        if verbose:
            print(f"[Step 6] Context length: {len(context)} chars")
        
        return context
    
    def answer(self, question: str, verbose: bool = False) -> str:
        """Full pipeline: retrieve and answer."""
        context = self.retrieve_context(question, verbose=verbose)
        
        if verbose:
            print(f"\n--- CONTEXT PASSED TO LLM ---\n{context[:2000]}")
            if len(context) > 2000:
                print("... [truncated in verbose output]")
        
        answer = generate_answer(self.answer_llm, question, context)
        return answer