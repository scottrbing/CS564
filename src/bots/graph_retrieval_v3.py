import json
import re
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


#Entity Extraction

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
  BAD aliases: "Layton Williams" -> ["Williams"] (too generic)
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
        print(f"[WARN] Failed to parse entity extraction: {text}")
        return []


# Fuzzy Entity Matching
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


# Subgraph Traversal

# Standard traversal for non-Publication nodes
ENTITY_TRAVERSAL_CYPHER = """
MATCH (anchor)
WHERE anchor.id = $node_id

OPTIONAL MATCH (anchor)-[r1]-(n1)
WHERE NOT n1:Document AND NOT n1:Date AND NOT n1:Publication

OPTIONAL MATCH (n1)-[r2]-(n2)
WHERE NOT n2:Document AND NOT n2:Date AND NOT n2:Publication
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

# NEW: For Publication nodes, don't traverse all REPORTED_BY edges.
# Instead, find entities that are connected to BOTH this publication 
# AND one of the other query entities. This filters out the noise.
PUBLICATION_TARGETED_CYPHER = """
MATCH (pub)
WHERE pub.id = $pub_id

// Find entities connected to this publication (either direction of REPORTED_BY)
MATCH (pub)-[:REPORTED_BY]-(entity)
WHERE NOT entity:Document AND NOT entity:Date
  AND entity.id IN $other_entity_ids

// Now get 1-hop from those shared entities (the interesting stuff)
OPTIONAL MATCH (entity)-[r]-(neighbor)
WHERE NOT neighbor:Document AND NOT neighbor:Date AND NOT neighbor:Publication
  AND neighbor.id <> pub.id

RETURN DISTINCT entity.id AS source, type(r) AS relationship, 
       neighbor.id AS target, 1 AS hop
LIMIT 50
"""

# NEW: When publication has no overlap with other entities,
# get its top REPORTED_BY connections as context anyway
PUBLICATION_DIRECT_CYPHER = """
MATCH (pub)
WHERE pub.id = $pub_id
MATCH (pub)-[:REPORTED_BY]-(entity)
WHERE NOT entity:Document AND NOT entity:Date
RETURN DISTINCT pub.id AS source, 'REPORTED_BY' AS relationship, 
       entity.id AS target, 1 AS hop
LIMIT 15
"""

# Source text: find Documents connected to a specific entity
SOURCE_TEXT_CYPHER = """
MATCH (d:Document)-[]->(anchor)
WHERE anchor.id = $node_id
RETURN DISTINCT d.text AS source_text
LIMIT 4
"""

# Source text filtered by BOTH publication and topic entity
# This finds articles from a specific publication about a specific topic
PUBLICATION_SOURCE_TEXT_CYPHER = """
MATCH (d:Document)-[:REPORTED_BY]->(pub),
      (d)-[:MENTIONS]->(entity)
WHERE pub.id = $pub_id AND entity.id = $entity_id
RETURN DISTINCT d.text AS source_text
LIMIT 3
"""

# Also try reverse direction of REPORTED_BY (since edges may be backwards)
PUBLICATION_SOURCE_TEXT_REVERSE_CYPHER = """
MATCH (d:Document)-[:MENTIONS]->(pub),
      (d)-[:MENTIONS]->(entity)
WHERE pub.id = $pub_id AND entity.id = $entity_id
RETURN DISTINCT d.text AS source_text
LIMIT 3
"""

# Bridging docs: documents mentioning multiple query entities
BRIDGING_DOCS_CYPHER = """
MATCH (d:Document)-[]->(e1), (d)-[]->(e2)
WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids
  AND e1.id <> e2.id
RETURN DISTINCT d.text AS source_text
LIMIT 8
"""


def traverse_subgraph(graph, matched_nodes: list[dict]) -> dict:
    """
    Traverse the graph with Publication-aware logic.
    
    For Publication nodes: use targeted traversal that only returns 
    entities shared with other query entities.
    For all other nodes: standard 2-hop traversal.
    """
    all_triples = []
    all_source_texts = []
    seen_triples = set()
    seen_texts = set()
    
    # Separate Publication nodes from other entities
    pub_nodes = [n for n in matched_nodes if 'Publication' in n.get('labels', [])]
    entity_nodes = [n for n in matched_nodes if 'Publication' not in n.get('labels', [])]
    
    # Get IDs
    entity_ids = [n["node_id"] for n in entity_nodes]
    pub_ids = [n["node_id"] for n in pub_nodes]
    
    # Traverse non-Publication entities
    for node in entity_nodes:
        nid = node["node_id"]
        
        try:
            results = graph.query(
                ENTITY_TRAVERSAL_CYPHER,
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
        
        # Get source texts for this entity
        try:
            docs = graph.query(SOURCE_TEXT_CYPHER, params={"node_id": nid})
            for row in docs:
                txt = row.get("source_text", "")
                if txt and txt not in seen_texts:
                    seen_texts.add(txt)
                    all_source_texts.append(txt)
        except Exception as e:
            print(f"[WARN] Source text failed for {nid}: {e}")
    
    # Traverse Publication nodes
    for pub_node in pub_nodes:
        pub_id = pub_node["node_id"]
        
        if entity_ids:
            # Find entities shared between publication and other query entities
            try:
                results = graph.query(
                    PUBLICATION_TARGETED_CYPHER,
                    params={"pub_id": pub_id, "other_entity_ids": entity_ids}
                )
                for row in results:
                    if row["target"] is None:
                        continue
                    triple_key = (row["source"], row["relationship"], row["target"])
                    if triple_key not in seen_triples:
                        seen_triples.add(triple_key)
                        all_triples.append({
                            "source": row["source"],
                            "relationship": row["relationship"],
                            "target": row["target"],
                            "hop": row["hop"],
                            "anchor": pub_id
                        })
            except Exception as e:
                print(f"[WARN] Publication targeted traversal failed for {pub_id}: {e}")
            
            # Get source texts from this publication about query entities
            for eid in entity_ids:
                try:
                    docs = graph.query(
                        PUBLICATION_SOURCE_TEXT_CYPHER,
                        params={"pub_id": pub_id, "entity_id": eid}
                    )
                    for row in docs:
                        txt = row.get("source_text", "")
                        if txt and txt not in seen_texts:
                            seen_texts.add(txt)
                            all_source_texts.insert(0, txt)  # Prepend — most relevant
                except:
                    pass
                
                # Try reverse direction also
                try:
                    docs = graph.query(
                        PUBLICATION_SOURCE_TEXT_REVERSE_CYPHER,
                        params={"pub_id": pub_id, "entity_id": eid}
                    )
                    for row in docs:
                        txt = row.get("source_text", "")
                        if txt and txt not in seen_texts:
                            seen_texts.add(txt)
                            all_source_texts.insert(0, txt)
                except:
                    pass
        else:
            # Publication is the only entity — get its top connections
            try:
                results = graph.query(
                    PUBLICATION_DIRECT_CYPHER,
                    params={"pub_id": pub_id}
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
                            "anchor": pub_id
                        })
            except Exception as e:
                print(f"[WARN] Publication direct traversal failed: {e}")
            
            # Get any source texts
            try:
                docs = graph.query(SOURCE_TEXT_CYPHER, params={"node_id": pub_id})
                for row in docs:
                    txt = row.get("source_text", "")
                    if txt and txt not in seen_texts:
                        seen_texts.add(txt)
                        all_source_texts.append(txt)
            except:
                pass
    
    # Bridging docs
    all_ids = entity_ids + pub_ids
    if len(all_ids) >= 2:
        try:
            bridge_docs = graph.query(
                BRIDGING_DOCS_CYPHER,
                params={"entity_ids": all_ids}
            )
            for row in bridge_docs:
                txt = row.get("source_text", "")
                if txt and txt not in seen_texts:
                    seen_texts.add(txt)
                    all_source_texts.insert(0, txt)
        except Exception as e:
            print(f"[WARN] Bridging doc fetch failed: {e}")
    
    # Also add REPORTED_BY triples connecting entities to publications so the LLM knows which publication covers which entity
    for pub_node in pub_nodes:
        pub_id = pub_node["node_id"]
        for eid in entity_ids:
            # Add explicit triple: entity was reported by publication
            triple_key_fwd = (eid, "REPORTED_BY", pub_id)
            triple_key_rev = (pub_id, "REPORTED_BY", eid)
            if triple_key_fwd not in seen_triples and triple_key_rev not in seen_triples:
                seen_triples.add(triple_key_fwd)
                all_triples.append({
                    "source": eid,
                    "relationship": "REPORTED_BY",
                    "target": pub_id,
                    "hop": 1,
                    "anchor": pub_id
                })
    
    return {
        "triples": all_triples,
        "source_texts": all_source_texts[:10]
    }


# Cross-Entity Relevance Filtering
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
        filtered = direct_triples[:40]
    else:
        MAX_TOTAL = 60
        filtered = key_triples[:10]
        remaining = MAX_TOTAL - len(filtered)
        filtered.extend(bridging_triples[:min(30, remaining)])
        remaining = MAX_TOTAL - len(filtered)
        filtered.extend(direct_triples[:min(20, remaining)])
    
    return filtered


# Context Formatting

def format_graph_context(subgraph: dict, matched_nodes: list[dict]) -> str:
    parts = []
    filtered_triples = filter_triples_by_relevance(subgraph["triples"], matched_nodes)
    anchor_ids = set(n["node_id"] for n in matched_nodes)
    
    if filtered_triples:
        parts.append("=== KNOWLEDGE GRAPH FACTS ===")
        for t in filtered_triples:
            src, rel, tgt = t["source"], t["relationship"], t["target"]
            if src in anchor_ids and tgt in anchor_ids:
                parts.append(f"[KEY] ({src}) -[{rel}]-> ({tgt})")
            else:
                parts.append(f"({src}) -[{rel}]-> ({tgt})")
    
    if subgraph["source_texts"]:
        parts.append("\n=== SUPPORTING DOCUMENT EXCERPTS ===")
        for i, txt in enumerate(subgraph["source_texts"], 1):
            truncated = txt[:800] + "..." if len(txt) > 800 else txt
            parts.append(f"\n[Excerpt {i}]\n{truncated}")
    
    if not parts:
        return "No relevant information found in the knowledge graph."
    
    return "\n".join(parts)


ANSWER_GENERATION_PROMPT = """You are a QA system answering questions using ONLY the context provided below.
 
The context contains:
1. KNOWLEDGE GRAPH FACTS: structured triples showing entity relationships.
   [KEY] triples directly connect entities mentioned in the question.
2. SUPPORTING DOCUMENT EXCERPTS: raw text from source articles.
 
CRITICAL REASONING RULES:
 
For Yes/No COMPARISON questions ("Does article A suggest X while article B suggests Y?"):
- These questions ask whether two claims are BOTH true based on the evidence.
- If the excerpts support EITHER claim, lean toward "Yes" — the question is asking 
  whether the articles present these perspectives, and having evidence for at least 
  one side with no contradicting evidence for the other typically means Yes.
- Only answer "No" if you find SPECIFIC evidence that directly CONTRADICTS one of 
  the claims made in the question.
 
For Yes/No CONSISTENCY questions ("Was the reporting consistent?" / "Was there a change?"):
- If multiple excerpts from the same or related sources discuss the same topic in 
  a similar tone or direction, the reporting IS consistent — answer "Yes".
- If excerpts show clearly DIFFERENT conclusions or contradictory claims about the 
  same topic, there WAS a change — consider which answer the question expects.
- Do NOT say "Unknown" just because you cannot verify exact publication dates. 
  Focus on whether the CONTENT of the excerpts is consistent or contradictory.
 
For INFERENCE questions ("Who/What/Which...?"):
- Follow the chain of relationships in the graph facts to identify the answer.
- Verify with document excerpts when possible.
 
GENERAL RULES:
- Only respond "Unknown" if the context contains NO relevant information at all.
  If you see excerpts discussing the topic, you MUST give a Yes/No or entity answer.
- Be concise: single entity name, date, short phrase, or Yes/No.
- Do NOT explain your reasoning.
 
Context:
{context}
 
Question: {question}"""


def generate_answer(llm: ChatOpenAI, question: str, context: str) -> str:
    response = llm.invoke([
        HumanMessage(content=ANSWER_GENERATION_PROMPT.format(
            context=context, question=question
        ))
    ])
    return response.content.strip()


# MAIN retriever
class GraphRAGRetriever:
    def __init__(self, neo4j_graph, openai_api_key: str, 
                 extraction_model: str = "gpt-4o-mini",
                 answer_model: str = "gpt-4o-mini"):
        self.graph = neo4j_graph
        self.extraction_llm = ChatOpenAI(
            temperature=0, model=extraction_model, api_key=openai_api_key
        )
        self.answer_llm = ChatOpenAI(
            temperature=0, model=answer_model, api_key=openai_api_key
        )
    
    def retrieve_context(self, question: str, verbose: bool = False) -> str:
        entities = extract_entities(self.extraction_llm, question)
        if verbose:
            print(f"[Step 1] Extracted entities: {json.dumps(entities, indent=2)}")
        
        if not entities:
            return "No entities could be extracted from the question."
        
        matched_nodes = match_entities_in_graph(self.graph, entities, max_per_entity=2)
        if verbose:
            for m in matched_nodes:
                print(f"  [Step 2] Matched: '{m['node_id']}' "
                      f"(via '{m['matched_on']}', tier={m['match_tier']}, "
                      f"coverage={m['name_coverage']:.2f}) "
                      f"labels={m['labels']}")
        
        if not matched_nodes:
            return "No matching entities found in the knowledge graph."
        
        subgraph = traverse_subgraph(self.graph, matched_nodes)
        if verbose:
            print(f"[Step 3] Raw: {len(subgraph['triples'])} triples, "
                  f"{len(subgraph['source_texts'])} source texts")
        
        context = format_graph_context(subgraph, matched_nodes)
        if verbose:
            filtered = filter_triples_by_relevance(subgraph["triples"], matched_nodes)
            print(f"[Step 4] After filtering: {len(filtered)} triples "
                  f"(from {len(subgraph['triples'])} raw)")
            print(f"[Step 5] Context length: {len(context)} chars")
        
        return context
    
    def answer(self, question: str, verbose: bool = False) -> str:
        context = self.retrieve_context(question, verbose=verbose)
        
        if verbose:
            print(f"\n--- CONTEXT PASSED TO LLM ---\n{context[:2000]}")
            if len(context) > 2000:
                print("... [truncated in verbose output]")
        
        answer = generate_answer(self.answer_llm, question, context)
        return answer
