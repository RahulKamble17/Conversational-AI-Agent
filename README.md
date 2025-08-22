--------------Conversational AI ChatBot-----------------------------

A skeleton or sample version of my use case. It uses RAG (PGVector), AWS Opus for OCR functions, relevancy check & summarization, FastApi to use with UI

Purpose: Support Engineers use SOP documents to resolve issues, so we need a chatbot to help them find the answers quickly as they are lazy and use Google & copilot instead (Actual customer statements lol)


------------------Usage Instructions--------------------------------
1. Create a venv install requiremts.txt
2. Install Postgres 13.20 or > . Create the PGVector extension.
3. You'll need AWS Bedrock access to run this code or modify with LLM of ur choice
4. Create a dir "pdf_dir" with all you pdfs.
5. Run the commented main function for ingesting PDFs
6. Uncomment it again cuz I was lazy to give you a functional code xD
7. Run the query search to get answers to any of ur questions from any uploaded PDF. (Its work in progress this is only a sample)
