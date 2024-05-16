# PDF-GPT: Intelligent PDF Interaction Platform

## Introduction to PDF-GPT 🤖📎

PDF-GPT represents a sophisticated platform seamlessly integrating artificial intelligence and document analysis, elevating the nuances of your PDF exploration. Upon uploading your document, a gateway to insightful conversations and revelations is unlocked. This innovative platform leverages the streamlit library, langchain, and the openai API to deliver a refined user experience.

## Getting Started

1. **Create a Python Virtual Environment**
   ```bash
   python -m venv venv
   ```
2. Activate the Virtual Environment
   - On Windows
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux
     ```bash
     source venv/bin/activate
     ```
3. Install requirements
   ```bash
   pip install -r requirements.txt
   ```
4. Create an '.env' file
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```
5. Run the app
   ```bash
   streamlit run app.py
   ```

## lm studio chat

I also added a script that makes it possible to interact with any open source model (freely available from huggingface and supplied to lm studio). [check out this link](https://lmstudio.ai/docs/local-server)
The script is called `chat.py` and can be run with the following command:

```bash
python3 chat.py
```
