# From https://blog.llamaindex.ai/multi-modal-rag-621de7525fea

## Multi-modal LLM
# Unlike our default LLM class, which has standard completion/chat endpoints,
# the multi-modal model (MultiModalLLM) can take in both image and text as input.
from llama_index.multi_modal_llms import OpenAIMultiModal
from llama_index import SimpleDirectoryReader

image_documents = SimpleDirectoryReader(local_directory).load_data()

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=300
)
response = openai_mm_llm.complete(
    prompt="what is in the image?", image_documents=image_documents
) 

## Multi-Modal Embeddings
# MultiModalEmbedding base class that can embed both text and images. 
# It contains all the methods as our existing embedding models (subclasses BaseEmbedding ) but also exposes get_image_embedding

## Multi-Modal Indexing and Retrieval
# MultiModalVectorIndex that can index both text and images into 
# underlying storage systems — specifically a vector database and docstore.
# this new index can store both text and image documents

documents = SimpleDirectoryReader("./mixed_wiki/").load_data()

# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_mm_db")

text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)
storage_context = StorageContext.from_defaults(vector_store=text_store)

# Create the MultiModal index
index = MultiModalVectorStoreIndex.from_documents(
    documents, storage_context=storage_context, image_vector_store=image_store
)

## Ex 1) Retrieval Augmented Captioning
retriever_engine = index.as_retriever(
    similarity_top_k=3, image_similarity_top_k=3
)
# retrieve more information from the GPT4V response
retrieval_results = retriever_engine.retrieve(query_str)

## Ex 2) Multi-Modal RAG Querying
from llama_index.query_engine import SimpleMultiModalQueryEngine

query_engine = index.as_query_engine(
    multi_modal_llm=openai_mm_llm,
    text_qa_template=qa_tmpl
)

query_str = "Tell me more about the Porsche"
response = query_engine.query(query_str)
