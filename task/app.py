import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY


SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """Initialize the RAG system"""
        print("Initializing Microwave Manual RAG System...")
        if os.path.exists("microwave_faiss_index"):
            print("Loading existing FAISS index...")
            vectorstore = FAISS.load_local(
                folder_path="microwave_faiss_index",
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating new FAISS index...")
            if not os.path.exists("task/microwave_manual.txt"):
                print("Error: microwave_manual.txt file not found.")
                raise FileNotFoundError("microwave_manual.txt file is missing.")
            vectorstore = self._create_new_index()
        
        return vectorstore

    def _create_new_index(self) -> VectorStore:
        print("Loading text document...")
        # 1. Create Text loader
        loader = TextLoader(file_path="task/microwave_manual.txt", encoding="utf-8")

        # 2. Load documents with loader
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")

        # 3. Create RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."]
        )

        # 4. Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from the document.")

        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk.page_content.strip():  # Check if chunk is not empty
                valid_chunks.append(chunk)
                print(f"Chunk {i + 1}: {chunk.page_content}")
            else:
                print(f"Chunk {i + 1} is empty and will be skipped.")

        print(f"{len(valid_chunks)} valid chunks will be added to the FAISS index.")

        # 5. Create vectorstore from valid chunks
        vectorstore = FAISS.from_documents(valid_chunks, self.embeddings)

        # 6. Save indexed data locally with index name "microwave_faiss_index"
        vectorstore.save_local("microwave_faiss_index")

        # 7. Return created vectorstore
        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=1.3) -> str:
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to x.
        """
        print(f"{'=' * 100}\nSTEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(f"Searching for top {k} most relevant chunks with similarity score {score}:")

        # TODO:
        #  Make similarity search with relevance scores`:
        try:
            results = self.vectorstore.similarity_search_with_score(
                  query=query,
                  k=k,
                  score_threshold=score
            )
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return ""

        if not results:
            print("No relevant context found for the query.")
            return ""

        print(f"Found {len(results)} results")

        context_parts = []
        # TODO:
        #  Iterate through results and:
        for i, (doc, doc_score) in enumerate(results):
            print(f"\n--- Result {i + 1} ---")
            print(f"Score: {doc_score}")
            print(f"Content: {doc.page_content}")
            context_parts.append(doc.page_content)
        #       - add page content to the context_parts array
        #       - print result score
        #       - print page content

        retrieved_context = "\n\n".join(context_parts)
        print(f"\nFinal Retrieved Context:\n{retrieved_context}")
        print("=" * 100)
        return retrieved_context

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"\nSTEP 2: AUGMENTATION\n{'-' * 100}")

        if not context:
            print("Warning: No context provided for augmentation.")

        #TODO: Format USER_PROMPT with context and query
        augmented_prompt = USER_PROMPT.format(context=context, query=query)

        print(f"Augmented Prompt:\n{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\nSTEP 3: GENERATION\n{'-' * 100}")

        # TODO:
        #  1. Create messages array with such messages:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]
        #       - System message from SYSTEM_PROMPT
        #       - Human message from augmented_prompt
        #  2. Invoke llm client with messages
        response = self.llm_client.generate(messages=[messages])

        # Extract the text from the first generation in the response
        if response.generations and response.generations[0]:
            generation = response.generations[0][0]
            print("Generated text:", generation.text)
            return generation.text
        else:
            print("No generations found in the response.")
            return "Error: No response generated."

def main(rag: MicrowaveRAG):
    print("Microwave RAG Assistant")

    while True:
        try:
            # Get user input
            user_question = input("\n> ").strip()
        except EOFError:
            # Handle end of file (for piped input)
            print("\nExiting the chat. Goodbye!")
            break
        
        # 5. If user message is `exit` then stop the loop
        if user_question.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break

        #TODO:
        # Step 1: make Retrieval of context
        context = rag.retrieve_context(query=user_question, k=4, score=1.3)
        # Step 2: Augmentation
        augmented_prompt = rag.augment_prompt(query=user_question, context=context)
        # Step 3: Generation
        rag.generate_answer(augmented_prompt=augmented_prompt)


main(
    MicrowaveRAG(
        # TODO:
        #  1. pass embeddings:
        AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small-1",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY)
        ),
        #       - AzureOpenAIEmbeddings
        #       - deployment is the text-embedding-3-small-1 model
        #       - azure_endpoint is the DIAL_URL
        #       - api_key is the SecretStr from API_KEY
        #  2. pass llm_client:
        AzureChatOpenAI(
            temperature=0.0,
            azure_deployment="gpt-4o",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            api_version=""
        )
        #       - AzureChatOpenAI
        #       - temperature is 0.0
        #       - azure_deployment is the gpt-4o model
        #       - azure_endpoint is the DIAL_URL
        #       - api_key is the SecretStr from API_KEY
        #       - api_version=""
    )
)