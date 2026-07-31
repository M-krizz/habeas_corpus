from neo4j import GraphDatabase

URI = "neo4j+s://0f2c4154.databases.neo4j.io"
USERNAME = "0f2c4154"
PASSWORD = "aLWh9a-fQHP4BRcYPKnNSQpPGuXmiZVLodvrrnPH0Fw"

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)

try:
    driver.verify_connectivity()
    print("✅ Connected successfully!")

    with driver.session() as session:
        result = session.run(
            "CREATE (n:Test {name:'Hello Neo4j'}) RETURN n.name AS name"
        )

        for record in result:
            print(record["name"])

except Exception as e:
    print(e)

finally:
    driver.close()