# EEEGPT v1.0: AI Assistant for Electrical and Electronic Engineering

EEEGPT is an AI-powered assistant designed to help students and professionals in Electrical and Electronic Engineering (EEE). It leverages advanced language models and vector search to provide accurate, context-based answers from authoritative EEE textbooks.

## Features

- **Contextual Q&A:** Answers questions using content from a trusted EEE textbook PDF.
- **Beginner-Friendly:** Explains concepts in simple language, with technical terms defined.
- **Technical Support:** Provides step-by-step reasoning, LaTeX equations, and code snippets (e.g., SPICE netlists) where relevant.
- **Source Transparency:** Optionally shows source document snippets for each answer.
- **Interactive UI:** Built with Streamlit for a conversational chat experience.

## Project Structure

```
.
├── connect_memory_with_llm.py
├── create_memory_for_llm.py
├── eeegpt_ui.py
├── test.py
├── requirements.txt
├── README.md
├── data/
│   └── a-textbook-of-electrical-technology-volume-i-basic-electrical-engineering-b-l-theraja.pdf
└── vectorstore/
    └── db_faiss/
        ├── index.faiss
        └── index.pkl
```

## How It Works

1. **Document Loading:** Loads and splits the textbook PDF into manageable text chunks.
2. **Embedding Generation:** Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` to create vector embeddings.
3. **Vector Store:** Stores embeddings in a FAISS vector database for efficient retrieval.
4. **LLM Integration:** Uses Groq's Llama 3 model via API for generating answers.
5. **Retrieval-Augmented Generation:** Retrieves relevant context from the vector store and feeds it to the LLM for accurate, context-aware responses.

## Setup Instructions

### 1. Clone the Repository

```sh
git clone https://github.com/Nobi004/eeegpt.git
cd eeegpt
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Prepare the Data

- Place your EEE textbook PDF(s) in the `data/` directory.

### 4. Create the Vector Store

Run the following script to process the PDF and build the FAISS vector database:

```sh
python create_memory_for_llm.py
```

### 5. Launch the Streamlit UI

```sh
streamlit run eeegpt_ui.py
```

## Usage

- Ask any EEE-related question in the chat interface.
- The assistant will answer using only the information from the provided textbook context.
- If the context is insufficient, it will let you know.

## Customization

- **Add More PDFs:** Place additional EEE PDFs in the `data/` folder and rerun `create_memory_for_llm.py`.
- **Prompt Engineering:** Modify the prompt template in `eeegpt_ui.py` to adjust the assistant's behavior.
- **Model Settings:** Change the LLM or embedding model as needed in the code.

## Troubleshooting

- Ensure all dependencies are installed.
- Make sure the FAISS database exists at `vectorstore/db_faiss/`.
- Verify your Groq API key is valid and set in the code.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [HuggingFace Transformers](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [Groq API](https://console.groq.com/)

## License

This project is for educational and research purposes only. Please ensure you have the rights to use any PDF content you provide.

---

*Made by Md. Mahmudun Nobi*
