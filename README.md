<center><h1>Agent-Centaur</h1></center>
<center><p>An AI Agent</p></center>
<center><img src="agent_centaur_logo.jpeg" width="300" height="300"></center>


##### 
# What is Agent-Centaur
    - A 'Subgraph ReAct' Agent architecture
    - Include agents such as ... 
        - Multi-agent RAG
        - Basic Summarization
        - Storm Researcher (not current integrated)
        - Timeline Extractor (using Map Extract)

# What is a Subgraph ReAct Agent
 - Subgraph ReAct agent is a ReAct agent that has subgraphs at it's disposal instead of, or in addition to, tools.
 - Each subgraph is a standalone agent that can be leveraged in the ReAct flow.

# Usage
 - Step 1, get documents and load then to the docs folder
 - Step 2, update the query parameter for search_and_scrape function
 - Step 3, run 'python ingestion.py'
 - Step 4, modify app.py with desired question
 - step 5, run 'python app.py'

