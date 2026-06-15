import streamlit as st
from rag_pipeline import process_pdf, build_vectorstore, build_qa_chain
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG Document Q&A")

if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Extracting, chunking, embedding..."):
            chunks = process_pdf(uploaded_file)
            vectorstore = build_vectorstore(chunks)
            st.session_state.chain = build_qa_chain(vectorstore)
            st.session_state.chat_history = []
        st.success(f"Done! {len(chunks)} chunks indexed.")

if st.session_state.chain:
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    if question := st.chat_input("Ask a question about your document..."):
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain({
                    "input": question,
                    "chat_history": st.session_state.chat_history
                })
                answer = result["answer"]
                sources = result["context"]

            st.write(answer)

            with st.expander("📚 Source Chunks"):
                for i, doc in enumerate(sources):
                    page = doc.metadata.get('page', '?')
                    st.markdown(f"**Chunk {i+1}** (Page {page})")
                    st.caption(doc.page_content[:300] + "...")

        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])

else:
    st.info("👈 Upload a PDF from the sidebar to get started.")