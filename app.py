import os
import tempfile
import streamlit as st

# Document Loaders & Splitters
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store & Embeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# RAG Chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Document Q&A Assistant", layout="wide")
st.title("📚 Document & Book Q&A Assistant")

# Sidebar for API Configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    st.markdown("---")
    st.write("Upload a PDF or DOCX file to ask questions about its content.")

if not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to proceed.")
    st.stop()

# Helper function to process uploaded files
def process_file(uploaded_file):
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    if file_ext == ".pdf":
        loader = PyPDFLoader(tmp_file_path)
    elif file_ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(tmp_file_path)
    else:
        st.error("Unsupported file format!")
        return None

    documents = loader.load()
    os.remove(tmp_file_path)  # Cleanup temporary file
    return documents

# File Upload Section
uploaded_file = st.file_uploader("Upload your PDF or Word document", type=["pdf", "docx"])

if uploaded_file:
    with st.spinner("Processing document and generating vector embeddings..."):
        docs = process_file(uploaded_file)
        
        if docs:
            # 1. Split document into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)

            # 2. Store in local vector store
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # 3. Create RAG Prompt Template
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise.\n\n"
                "{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # 4. Build Chain
            llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0)
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            st.success(f"Document processed successfully! Total chunks created: {len(splits)}")

            # Session state for chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display past messages
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            # User Question Input
            user_question = st.chat_input("Ask a question about your document...")
            if user_question:
                st.session_state.messages.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.write(user_question)

                with st.chat_message("assistant"):
                    with st.spinner("Searching document for answers..."):
                        response = rag_chain.invoke({"input": user_question})
                        answer = response["answer"]
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})