import streamlit as st 
import requests
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional


HUGGING_FACE_REPO_ID = "llama3-8b-8192"
GROQ_API_KEY = "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"
DB_FAISS_PATH = "vectorstore/db_faiess"
@st.cache_resource

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db=FAISS.loaad_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

class GroqLLM(LLM):
    model_name: str = "llama3-8b-8192"  # Default model
    temperature: float = 0.5
    max_tokens: int = 800
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if stop:
            data["stop"] = stop
            
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error from Groq API: {response.text}")
            
        return response.json()["choices"][0]["message"]["content"]
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

def main():
    st.title("Welcme to EEGPT")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Ask your quiestion herer:")


    if prompt:

        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user","content":prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        

        # response = "Hi! EEEians! I am here to help you with your queries. Please ask me anything related to your studies."
        # st.chat_message('assistant').markdown(response)

        try:
            # Create embedding model
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Load FAISS database
            DB_FAISS_PATH = "vectorstore/db_faiss"
            db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            
            # Initialize the Groq LLM
            llm = GroqLLM(
                model_name="llama3-8b-8192",  # Using Llama 3 from Groq
                temperature=0.5,
                max_tokens=800
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response['source_documents']
            result_to_show=result+"\nSource Documents:\n"+str(source_documents)
        
            print("RESULT: ", response["result"])
            print("\nSOURCE DOCUMENTS:")
            for i, doc in enumerate(response["source_documents"]):
                print(f"\nDocument {i+1}:")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Content: {doc.page_content[:150]}...")  # Show first 150 chars
            
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})
        except Exception as e:
            print(f"An error occurred: {e}")
            print("\nTroubleshooting suggestions:")
            print("1. Check if you have installed all required packages: pip install requests faiss-cpu langchain-huggingface")
            print("2. Ensure the FAISS database exists at path:", DB_FAISS_PATH)
            print("3. Make sure your Groq API key is valid")
            st.error(f"Error: {str(e)}")
            



if __name__ == "__main__":
    main()