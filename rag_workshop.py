ASTRA_DB_API_ENDPOINT = "https://78e5fc41-a92d-43ee-be98-1dc613a9792d-us-east1.apps.astra.datastax.com"
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:TBIwGbRWytSBEsyUBxrvstwz:717221911013217a4cf5d8944df761ead1e05f9158def21f75e86a1f4fc9796c"


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import AstraDB
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.schema.runnable import RunnableMap
import requests

def load_chat_model():
    # parameters for ollama see: https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.ollama.ChatOllama.html
    # num_ctx is the context window size
    return ChatOllama(model="mistral:latest", num_ctx=18192)

docs = [
    Document(
        page_content=f"""
Next Step:
08/16/2023 : Review partner information updates and update opportunity details. 8/17(LR) - connecting with Partner to offer co-sell support

Next Step History:
null;08/16/2023 : Review partner information updates and update opportunity details.;08/16/2023 : Review partner information updates and update opportunity details. 8/17(LR) - connecting with Partner to offer co-sell support
""",
        metadata={"customer_id": 'CUS100', "partner_id": 'AWS', "opportunity_id": 'WS-7202838a', "customer_name": 'Teradyne, Inc.' },
    ),
    Document(
        page_content=f"""
Action Items:
From Autumn, send recording of last call and our discussed inputs from demo 8/28. Ramesh will provide to Caroline by early next week (of 8/11).
""",
        metadata={"customer_id": 'CUS100', "partner_id": 'AWS', "opportunity_id": 'WS-8a038b8a', "customer_name": 'Teradyne, Inc.' },
    ),
    Document(
        page_content=f"""
Action Items:
Joint sync set for 9/7. Enablement session to follow + in person account mapping. Caroline / Michael to begin coordinating. EAI presence
""",
        metadata={"customer_id": 'CUS100', "partner_id": 'AWS', "opportunity_id": 'WS-8a3b0348', "customer_name": 'Teradyne, Inc.' },
    ),
    Document(
        page_content=f"""
Action Items:
From Caroline, user community engaged to respond to questions. @Dataiku - How can we get initial data from user community/pull together PoV for client? Action (Asan/Ken (sp?)): In-person outreach to Deloitte users and follow-up to 5 responses received.
""",
        metadata={"customer_id": 'CUS100', "partner_id": 'AWS', "opportunity_id": 'WS-8a7128a3', "customer_name": 'Teradyne, Inc.' },
    ),
    Document(
        page_content=f"""
Propsal did not go thru. No budget Left. Negative.
""",
        metadata={"customer_id": 'CUS101', "partner_id": 'AWS', "opportunity_id": 'WS-8a7128a4', "customer_name": 'Teradyne, Inc.' },
    ),
]


vector = AstraDB.from_documents(docs, embeddings,collection_name="workspan_mistral", api_endpoint=ASTRA_DB_API_ENDPOINT, token=ASTRA_DB_APPLICATION_TOKEN)


# Define a retriever interface
retriever = vector.as_retriever()
# Define LLM
chat_model = load_chat_model()
# Define prompt template
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# Create a retrieval chain to answer questions
document_chain = create_stuff_documents_chain(chat_model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("Invoking the question")
response = retrieval_chain.invoke({"input": "What are the next steps?"})
print(response["answer"])
