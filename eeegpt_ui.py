import streamlit as st 
import requests
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
import base64

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
    with open("images/logo.png", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <h1 style='text-align: center;'>Welcome to EEEGPT v1.0</h1>
        <div style="display: flex; justify-content: flex-end;">
            <h3 style='text-align: left;'>Creator: Md. Mahmudun Nobi</h3>
            <img src="data:image/jpg;base64,{b64_string}" alt="Profile" style="
                width: 130px;
                height: 130px;
                object-fit: cover;
                border-radius: 50%;
                border: 3px solid #ddd;
                margin-top: 10px;
                margin-right: 10px;
                align:right;
            ">
        </div>
        <h4 style='text-align: center;'>Your AI Assistant for Electrical and Electronic Engineering study.</h4>
        """, unsafe_allow_html=True)
    st.markdown(
        """
        <h3 style='text-align: center;'>mdmnb435@gmail.com</h3>
        <h5 style='text-align: left;'>Still in development phase,If you have any suggession please give me feedback through mail.</h5>
        """,
        unsafe_allow_html=True
    )
    # st.title("Welcme to EEEGPT v1.0")
    # st.subheader("Made by Md. Mahmudun Nobi")
    # st.subheader("Your AI Assistant for Electrical and Electronic Engineering study.")
    # st.write("Ask me anything related to your studies, and I'll do my best to help you!")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Ask your quiestion herer:")


    if prompt:

        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user","content":prompt})

        # CUSTOM_PROMPT_TEMPLATE = """
        #             You are EEEGPT, a friendly AI assistant for electrical and electronic engineering (EEE), trained on a single EEE book. Your goal is to explain EEE concepts in a smooth, clear, and engaging way that any student—beginner or advanced—can easily understand. Follow these guidelines:

        #             1. Use simple, friendly language, like talking to a friend. Avoid technical jargon unless you explain it in plain words (e.g., 'voltage is like water pressure').
        #             2. If the context doesn’t have enough information, say: 'The context doesn’t give enough details to answer this question.'
        #             3. Break answers into 2–3 short, clear steps or points to make concepts easy to follow.
        #             4. Use fun, everyday examples or analogies (e.g., compare circuits to water pipes or a flashlight) to make ideas relatable.
        #             5. For equations, use LaTeX (e.g., $$ V = IR $$) and explain each part in simple terms (e.g., 'V is the push that moves electricity').
        #             6. If code is needed (e.g., SPICE), include it in a code block and describe it simply.
        #             7. If the question is unclear, briefly explain how you’re interpreting it before answering.
        #             8. Keep answers short (150–200 words) and include one real-world example to show how the concept is used (e.g., in a phone charger).
        #             9. Make the tone encouraging and fun to keep students motivated, but stay accurate and professional.

        #             Context: {context}
        #             Question: {question}

        #             Answer:
        #             """

        CUSTOM_PROMPT_TEMPLATE = """
                You are EEEGPT, an AI assistant for electrical and electronic engineering (EEE), trained on a single authoritative EEE book. Your goal is to provide clear, easy-to-understand, and accurate answers based only on the provided context. Follow these guidelines:

                1. Use simple, everyday language to explain concepts, as if teaching a beginner. Avoid jargon unless necessary, and define any technical terms.
                2. Answer only using the context. If the context doesn’t have enough information, say: 'The context does not provide enough information to answer this question.'
                3. Break down technical answers into clear, numbered steps or bullet points to make them easy to follow.
                4. Use examples or analogies from the context to make ideas relatable (e.g., compare circuits to water pipes if it helps).
                5. For equations, use LaTeX (e.g., $$ V = IR $$) and explain what each term means in plain words.
                6. If the question involves code (e.g., SPICE), provide it in a code block and describe what it does simply.
                7. If the question is unclear, briefly state your interpretation before answering.
                8. Keep answers concise (under 200 words) but include one practical example or application to show how the concept is used in real life.
                9. Ans questions using some keypoints and bold the important terms.
                10. Include a summary at the end of the answer to reinforce understanding.

                Context: {context}
                Question: {question}

                Answer:
                """
        #  """
        #     You are EEEGPT, a specialized AI for electrical and electronic engineering (EEE), trained on authoritative EEE texts. Your goal is to provide accurate, concise, and technically precise answers based solely on the provided context. Follow these guidelines:

        #     1. Answer directly using the context, avoiding speculation or external information.
        #     2. If the context lacks sufficient information, state: 'The provided context does not contain enough information to answer this question.'
        #     3. For technical questions, include step-by-step reasoning or calculations where applicable, using LaTeX for equations (e.g., $$ V = IR $$) or code blocks for snippets (e.g., SPICE netlists).
        #     4. If the question is ambiguous, briefly clarify the interpretation before answering.
        #     5. Tailor the response to the user's likely expertise level (inferred from the question's complexity) while maintaining technical accuracy.
        #     6. Provide relevant examples or applications from the context to enhance understanding, especially for practical EEE tasks like circuit design or power analysis.

        #     Context: {context}
        #     Question: {question}

        #     Answer:
        #     """
        
        

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
            result_to_show=result # +"\nSource Documents:\n"+str(source_documents)
        
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