from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
import time
import certifi
import tempfile

# load groq token variable 
groq_api_key = os.getenv("GROQ_API_KEY")

os.environ["SSL_CERT_FILE"] = certifi.where()

# load all langchain tools
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


# setup llm
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)  

# setup prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful and knowledgeable assistant. Answer the user's question using only the information provided in the <context> section below.
    <context>
    {context}
    </context>
    Question: {input}
    Instructions:
    - If the answer exists in the context, provide a clear, concise, and accurate response.
    - If the answer cannot be found in the context, reply with:
    "I'm sorry, I do not have enough information in the document to answer that."

    Only use the context provided. Do not make assumptions or use external knowledge.
    """
)

uploaded_files  = st.file_uploader("Upload your documents", type=['pdf', 'txt'], accept_multiple_files=True)

# store multiple file
loaders = [] 
if uploaded_files  and "vector_store" not in st.session_state:
    for upload_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(upload_file.read())
            temp_path = temp_file.name
    
        if upload_file.name.endswith(".pdf"):
            loaders.append(PyPDFLoader(temp_path)) 
        elif upload_file.name.endswith(".txt"):
            loaders.append(TextLoader(temp_path))
        else:
            st.error("Unsupported file type")
            st.stop()

    # handle vector embedding for documents
    def create_vector_embedding():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        all_docs = []
        for loader in loaders:
            loaded = loader.load()
            if loaded:
                all_docs.extend(loaded) 
        text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        final_documents = text_spliter.split_documents(all_docs)
        if not final_documents:
            st.error("Document splitting failed. No chunks were generated.")
            st.stop()
        st.session_state.vector_store = Chroma.from_documents(final_documents, embeddings)

    with st.spinner("Embedding.... Please wait some time"):
        try: 
            create_vector_embedding()
            st.success("Document embedded and ready for question answering.")
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            st.stop()

# init chat history
if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {
            "role": "assistant",
            "content": "Hi there! I’m your Support Assistant. Feel free to ask me anything — I’m here to help!."
        }
    ]

for message in st.session_state.messages:
    st.chat_message(message['role']).write(message['content'])

# response generator
def generate_response(user_question):
    retriever = st.session_state.vector_store.as_retriever()
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    response = rag_chain.invoke({"input": user_question})
    # extract the answer safely
    return response.get("answer", "I'm sorry, I do not have enough information in the document to answer that.")

question = st.chat_input("Ask me any question")

if question and "vector_store" in st.session_state:
    if question.strip():
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Analyzing Response..."):
            final_answer = generate_response(question)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.chat_message("assistant").write(final_answer)
    else:
        st.info("Please provide a valid question")
