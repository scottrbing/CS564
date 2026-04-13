from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "<cuurent_passowrd>")
)

with driver.session() as session:
    result = session.run("RETURN 1 AS test")
    print(result.single())