import streamlit as st
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from groqapi import query_groq

# Load legal models and embeddings
@st.cache_resource
def load_legal_models():
    # Load fine-tuned Hugging Face legal model
    qa_pipeline = pipeline("question-answering", model="nlp/legal-finetuned-model")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/legal-bert")
    return qa_pipeline, embeddings

qa_pipeline, embeddings = load_legal_models()

# Load or create vector database for precedents
@st.cache_resource
def load_vector_db():
    # Assumes case precedents are preprocessed into embeddings
    return FAISS.load_local("case_precedents", embeddings)

db = load_vector_db()

# Streamlit UI
st.title("Legal Assistant")
st.sidebar.title("Features")
feature = st.sidebar.radio(
    "Choose a feature", ["Automated Document Drafting", "Legal Q&A", "Retrieve Legal Precedents"]
)

if feature == "Automated Document Drafting":
    st.header("Automated Document Drafting")
    document_type = st.selectbox("Select Document Type", ["Contract", "Will", "Power of Attorney"])
    inputs = {
        "Contract": ["Party Names", "Contract Terms", "Effective Date", "Termination Date"],
        "Will": ["Testator Name", "Beneficiaries", "Executor"],
        "Power of Attorney": ["Grantor Name", "Attorney Name", "Effective Date"]
    }
    user_inputs = {}
    for field in inputs[document_type]:
        user_inputs[field] = st.text_input(field)
    
    if st.button("Generate Document"):
        st.write("### Generated Document:")
        # Mock template rendering; integrate with document generation tools
        st.text(f"Drafted {document_type}: \n\n{user_inputs}")

elif feature == "Legal Q&A":
    st.header("Legal Q&A")
    query = st.text_input("Ask a legal question:")
    if st.button("Get Answer"):
        response = qa_pipeline({"question": query, "context": "Provide your legal context or database text here."})
        st.write("### Answer:")
        st.write(response["answer"])

elif feature == "Retrieve Legal Precedents":
    st.header("Retrieve Legal Precedents")
    case_query = st.text_input("Search for case precedents:")
    if st.button("Search"):
        retriever = RetrievalQA.from_chain_type(llm=qa_pipeline, retriever=db.as_retriever())
        results = retriever.run(case_query)
        st.write("### Relevant Precedents:")
        st.write(results)

# Footer
st.sidebar.info("Powered by Hugging Face, Groq, and LangChain.")
