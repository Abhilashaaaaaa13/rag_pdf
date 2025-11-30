import os
import streamlit as st
import base64
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# ---------------------- Environment ----------------------
def load_environment():
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ---------------------- Streamlit page ----------------------
def setup_streamlit_page():
    st.set_page_config(layout="wide")
    st.title("Conversational RAG over PDFs 🤖📚")

# ---------------------- Session chat ----------------------
def initialize_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def add_chat(role, content):
    st.session_state.messages.append({"role": role, "content": content})

# ---------------------- PDF processing ----------------------
def process_pdfs(files):
    documents = []
    extracted_text = []
    for i, uploaded_file in enumerate(files):
        temp_path = f"./temp_{i}.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)
        extracted_text.append("\n\n".join([d.page_content for d in docs]))
    return documents, extracted_text

def display_pdf_and_text(uploaded_files, extracted_text):
    with st.sidebar:
        st.markdown("### Uploaded PDFs")
        for i, file in enumerate(uploaded_files):
            st.markdown(f"**{file.name}**")
            file.seek(0)
            base64_pdf = base64.b64encode(file.read()).decode("utf-8")
            pdf_view = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="300"></iframe>'
            st.markdown(pdf_view, unsafe_allow_html=True)
            st.markdown("**Content Preview:**")
            st.text_area("Preview", extracted_text[i][:600], height=120)

# ---------------------- Free embeddings ----------------------
class LocalEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# ---------------------- Build RAG chain ----------------------
def build_rag_chain(documents, llm):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(documents)

    embeddings = LocalEmbeddings()
    vectorstore = Chroma.from_documents(
        splits, embeddings, persist_directory="./chroma_store"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def rag_chain_fn(inputs):
        query = inputs["question"]
        results = retriever.get_relevant_documents(query)
        combined_context = "\n\n".join([d.page_content for d in results])

        prompt = (
            "You are an AI assistant. Answer ONLY based on the context below.\n"
            "If the answer is not in the context, reply: 'I could not find the answer in the documents.'\n\n"
            f"Context:\n{combined_context}\n\nQuestion: {query}"
        )

        response = llm([HumanMessage(content=prompt)])

        # --- EXTRACT ONLY CLEAN TEXT ---
        try:
            # ChatGroq sometimes returns a list of dict-like objects
            first = response[0] if isinstance(response, list) else response
            # Access the 'content' attribute safely
            answer_text = getattr(first, "content", None) or first.get("content", str(first))
            # Remove any wrapper if accidentally returned
            if isinstance(answer_text, str) and answer_text.startswith("content="):
                start = answer_text.find("'") + 1
                end = answer_text.rfind("'")
                answer_text = answer_text[start:end]
            return answer_text.strip()
        except Exception:
            return str(response).strip()

    return rag_chain_fn

# ---------------------- Main ----------------------
def main():
    load_environment()
    setup_streamlit_page()
    initialize_chat_state()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter your GROQ API key:", type="password")
    if not api_key:
        st.warning("Please provide a valid GROQ API key to continue.")
        return

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

    st.info("Using free embeddings (SentenceTransformer) + GROQ LLM.")

    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        docs, extracted = process_pdfs(uploaded_files)
        display_pdf_and_text(uploaded_files, extracted)

        rag_chain = build_rag_chain(docs, llm)

        user_q = st.text_input("Ask your question:")
        if user_q:
            add_chat("user", user_q)
            with st.spinner("Searching in PDFs..."):
                answer = rag_chain({"question": user_q})  # <-- CLEAN TEXT ONLY

            add_chat("assistant", answer)

            st.subheader("Answer:")
            st.write(answer)  # ✅ only the answer text

            with st.expander("Chat History"):
                for msg in st.session_state.messages:
                    st.markdown(f"**{msg['role']}**: {msg['content']}")

if __name__ == "__main__":
    main()
