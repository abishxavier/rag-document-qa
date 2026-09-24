from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile, os
from dotenv import load_dotenv

load_dotenv()

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    os.unlink(tmp_path)
    return chunks


_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def build_vectorstore(chunks):
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    return vectorstore


def build_qa_chain(vectorstore, api_key=None):
    effective_api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not effective_api_key:
        raise ValueError("GROQ_API_KEY is not configured. Please provide an API key.")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=effective_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Prompt to rephrase follow-up questions using chat history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "rephrase it as a standalone question. Do not answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Prompt to answer using retrieved context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer the question using the context below. "
         "If you don't know, say you don't know.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_context(inp):
        question = inp["input"]
        chat_history = inp["chat_history"]

        # If there's chat history, rephrase the question first
        if chat_history:
            rephrase_chain = contextualize_prompt | llm | StrOutputParser()
            question = rephrase_chain.invoke({
                "input": question,
                "chat_history": chat_history
            })

        docs = retriever.invoke(question)
        return {"context": format_docs(docs), "docs": docs}

    def full_chain(inp):
        context_result = get_context(inp)
        answer = (qa_prompt | llm | StrOutputParser()).invoke({
            "input": inp["input"],
            "chat_history": inp["chat_history"],
            "context": context_result["context"]
        })
        return {
            "answer": answer,
            "context": context_result["docs"]
        }

    return full_chain