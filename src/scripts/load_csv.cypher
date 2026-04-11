// ---------------------
// CONSTRAINTS
// ---------------------
CREATE CONSTRAINT IF NOT EXISTS FOR (w:Word) REQUIRE w.text IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;

// ---------------------
// LOAD WORDS
// ---------------------
LOAD CSV WITH HEADERS FROM 'file:///words.csv' AS row
MERGE (w:Word {text: row.text});

// ---------------------
// LOAD CATEGORIES
// ---------------------
LOAD CSV WITH HEADERS FROM 'file:///categories.csv' AS row
MERGE (c:Category {name: row.name});

// ---------------------
// LOAD DOCUMENTS
// ---------------------
LOAD CSV WITH HEADERS FROM 'file:///documents.csv' AS row
MERGE (d:Document {id: row.id});

// ---------------------
// DOC → CATEGORY
// ---------------------
LOAD CSV WITH HEADERS FROM 'file:///doc_category.csv' AS row
MATCH (d:Document {id: row.doc_id})
MATCH (c:Category {name: row.category})
MERGE (d)-[:BELONGS_TO]->(c);

// ---------------------
// DOC → WORD
// ---------------------
LOAD CSV WITH HEADERS FROM 'file:///doc_word.csv' AS row
MATCH (d:Document {id: row.doc_id})
MATCH (w:Word {text: row.word})
MERGE (d)-[:CONTAINS]->(w);

// ---------------------
// WORD SIMILARITY
// ---------------------
LOAD CSV WITH HEADERS FROM 'file:///word_similarity.csv' AS row
MATCH (w1:Word {text: row.word1})
MATCH (w2:Word {text: row.word2})
MERGE (w1)-[:SIMILAR_TO {score: toFloat(row.score)}]->(w2);