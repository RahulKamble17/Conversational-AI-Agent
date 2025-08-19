import fitz 
#import camelot
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import markdown
from typing import List, Dict, Any
import os
import pandas as pd
import json
import psycopg2
import base64
from psycopg2 import OperationalError
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
import uvicorn
 
aws_llm = ChatBedrock(
    # Use the inference profile ARN here instead of the generic model ID
    model_id="Enter-Your-ID-Here",
    # **Explicitly specify the provider here**
    provider="anthropic",
    # Pass model-specific parameters via model_kwargs
    model_kwargs={
        "temperature": 0,
        "max_tokens": None,
    }
)
 
#------------------------EXTRACTION SECTION-----------------------------------------------------
def extract_text_from_page(doc, page_num):
    page = doc.load_page(page_num - 1)
    return page.get_text()
def extract_images_from_page(doc, page_num, output_dir):
    images_on_page = []
    page = doc.load_page(page_num - 1)
    # Create an images subdirectory if it doesn't exist
    images_dir = os.path.join(output_dir, f"page_{page_num}_images")
    os.makedirs(images_dir, exist_ok=True)
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_filename = f"image_{img_index+1}.{image_ext}"
        image_path = os.path.join(images_dir, image_filename)
        with open(image_path, "wb") as img_file:
            img_file.write(image_bytes)
        images_on_page.append(image_path)
    return images_on_page
#def extract_tables_from_page(pdf_path, page_num, output_dir):
#    tables_on_page = []
#    # Camelot's pages parameter expects a string
#    page_str = str(page_num)
#    # Extract only bordered tables using the 'lattice' flavor
#    tables = camelot.read_pdf(pdf_path, pages=page_str, flavor='lattice')
#    if not tables:
#        return tables_on_page
#    tables_dir = os.path.join(output_dir, f"page_{page_num}_tables")
#    os.makedirs(tables_dir, exist_ok=True)
#    for i, table in enumerate(tables):
#        df = table.df
#        csv_filename = f"table_{i+1}.csv"
#        csv_path = os.path.join(tables_dir, csv_filename)
#        df.to_csv(csv_path, index=False)
#        tables_on_page.append(csv_path)
#    return tables_on_page
##---------Main Function to Process the PDF Page by Page--------------
def extract_content_page_by_page(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' does not exist.")
        return None
    #----------Create a single output directory for all extracted content----------
    output_dir_base = "extracted_content_" + os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(output_dir_base, exist_ok=True)
    #----The final data structure to hold all extracted content-----
    all_extracted_content = []
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        print(f"Processing '{pdf_path}' with {num_pages} pages...")
        for page_num in range(1, num_pages + 1):
            print(f"\n--- Processing Page {page_num}/{num_pages} ---")
            page_data = {
                "page_num": page_num,
                "text": "",
                "tables": [],
                "images": []
            }
            #-----------------Extract Text-------------------
            page_data["text"] = extract_text_from_page(doc, page_num)
            #-----------------Extract Tables-----------------
            #page_data["tables"] = extract_tables_from_page(pdf_path, page_num, output_dir_base)
            #-----------------Extract Images-----------------
            page_data["images"] = extract_images_from_page(doc, page_num, output_dir_base)
            all_extracted_content.append(page_data)
        doc.close()
        print("\nPDF processing complete.")
        return all_extracted_content
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        return None
 
#--------------------------Base64 image---------------------------------
def get_image_as_base64(image_path):
    """Encodes a local image file to a base64 string with correct MIME type."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            # Determine MIME type based on file extension
            if image_path.lower().endswith(".png"):
                mime_type = "image/png"
            elif image_path.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            else:
                print(f"Warning: Unknown image type for {image_path}. Using application/octet-stream.")
                mime_type = "application/octet-stream"
            return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}.")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path} to Base64: {e}")
        return None
 
 
#------------------Extract Text from Images-----------------------------
def extract_text_from_image(image_path: str) -> str | None:
    print(f"Attempting to extract text from image (LLM call): {image_path}")
    try:
        base64_image_data = get_image_as_base64(image_path)
        if not base64_image_data:
            return None
 
        image_content_dict = {"type": "image_url", "image_url": {"url": base64_image_data}}
 
        response = aws_llm.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Extract all text from this image. Do not summarize or provide any additional commentary, just return the raw, complete text content from the image, preserving line breaks if possible. If no text is clearly visible, state 'NO_TEXT_DETECTED'."},
                        image_content_dict,
                    ]
                )
            ]
        )
        extracted_text = response.content.strip()
 
        if extracted_text == "NO_TEXT_DETECTED":
            print(f"No text detected in image: {image_path}")
            # Cache a specific marker for "NO_TEXT_DETECTED" to avoid future LLM calls
            return None
       
        print(f"Successfully extracted text from image: {image_path}")
       
        return extracted_text
    except Exception as e:
        print(f"Error processing image {image_path} for text extraction: {e}")
        return None
 
 
#-----------------------------EMBEDDING CREATION SECTION---------------------------------------------------
 
# Function to get embeddings for chunks
def get_embedding(input,type):
  try:
      embeddings = BedrockEmbeddings(
          model_id="Enter-Your-ID-Here",
          region_name="Enter-Region"
      )
  except Exception as e:
    print(f"Error initializing BedrockEmbeddings: {e}")
    return None
   
  if type=="chunks":
    embeddings_list=embeddings.embed_documents(texts=input)
    embeddings_list=[f'{embed}' for embed in embeddings_list]
    return embeddings_list
  elif type=="query":
    embedding=embeddings.embed_query(input)
    return embedding
 
 
#--------------------------Get CHUNKS From JSON-------------------------------------
def get_chunks_from_json(json_file_path):
    if not os.path.exists(json_file_path):
       print(f"Error: The file '{json_file_path}' does not exist.")
       return None
    with open(json_file_path, 'r') as f:
       extracted_data = json.load(f)
 
    #-----Loop through each page and access the content-----     
    if extracted_data:
        chunks=[]
        pages_content=extracted_data    
        for page_data in pages_content:
            chunk=""
            page_num = page_data['page_num']
            text = page_data['text']
            tables = page_data['tables']
            images = page_data['images']
            print(f"--- Processing Page {page_num} ---")
            if text:
                print(f"Text length: {len(text)} characters")
                chunk+=text+"\n"
            #text_embedding = my_text_embedding_function(text)
            if tables:
                print(f"Found {len(tables)} table(s) on this page.")
                combined_data = ""
                table_number = 1
                for table_path in tables:
                    table_path = os.path.abspath(table_path)
 
                    if table_path.endswith(".csv"):
                        df = pd.read_csv(table_path)
                        combined_data += f"\nTable {table_number}:\n"
                        for _, row in df.iterrows():
                            combined_data += ",".join(map(str, row.values)) + "\n"
                        table_number += 1
                chunk+="Following are tables as csv:\n"+combined_data+"\n"
            if images:
                print(f"Found {len(images)} image(s) on this page.")
                combined_img_txt=""
                img_number=1
                for image_path in images:
                    image_path = os.path.abspath(image_path)
                    img_txt=extract_text_from_image(image_path)
                    if img_txt is not None:
                        combined_img_txt+=f"\nImage {img_number};\n"+img_txt+"\n"
                chunk+="Follwing are text extracted from images/diagrams:\n"+combined_img_txt+"\n"
            chunks.append(chunk)
            print("-" * 25)
        return chunks
    else:
        print("Failed to process the JSON file.")
        return None
 
#------------------------Connections to PGVECTOR--------------------------------------------
#---------Connect to Postgres---------
def connect_to_postgres(host, database, user, password, port=5432):
    try:
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        print("Connection to PostgreSQL DB successful")
        return conn
    except OperationalError as e:
        print(f"Error occurred while connecting to Postgres: '{e}' ")
        return None
 
#----------Function to ingest into PGVECTOR------------
def ingest_into_pgvector(conn,chunks,embeddings,file_name,img_path="NULL"):
    try:
        cur=conn.cursor()
        print("\t\t---file name:  ",file_name,"\n\n")
        cur.execute(f"CREATE TABLE IF NOT EXISTS {file_name}(chunk TEXT,embedding vector(1024));")
        #-----Assemble values for bulk insertion------
        values_list=[]
        for i in zip(chunks,embeddings):
            values_list.append(i)
   
        values=",".join(map(str,values_list))
        #if img_path=="NULL":
        #    cur.execute(f"INSERT INTO {file_name} (chunk,embedding) VALUES {values}")
        #else:
        #    cur.execute(f"INSERT INTO {file_name} (chunk,embedding) VALUES {values}")
        cur.execute(f"INSERT INTO {file_name} (chunk,embedding) VALUES {values}")
        conn.commit()
        cur.close()
        return print(f"{file_name} inserted\n")
    except Exception as e:
        return print(f"Error while inserting: '{e}' ")
 
#-------Function to Query from VectorStore---------
def search_query(conn,file_name,user_query):
    try:
        print("\n\n\t--------Trying query embedd----------\n\n")
        user_qury_embedding=get_embedding(user_query,"query");
        print("\n\n\t--------Query embedded----------\n\n")
        cur=conn.cursor()
        cur.execute(f"SELECT chunk from {file_name} ORDER BY embedding <-> '{user_qury_embedding}' LIMIT 3; ")
        print("\n\n\t--------SELECTED data from PG Query----------\n\n")
        rows=cur.fetchall()
        docs=""
        chunk_cntr=1
        for row in rows:
            docs+=f"Chunk {chunk_cntr}:\n"+str(row[0])+"\n\n"
            chunk_cntr+=1
        return docs
        cur.close()
    except Exception as e:
        print("\n\n\t--------Error while Querying----------\n\n")
        return print(f"Error while querying: '{e}' ")
 
 
#-----------------------------ANSWER GENERATION WITH LLM SECTION----------------------------------------
 
def get_answer_from_llm(query,docs):
    try:
        print("\n\n\t--------Trying LLM call for answer summary----------\n\n")
        system_prompt_template = SystemMessagePromptTemplate.from_template(
                    "You are an expert summarization assistant. Summarize the following retrieved documents into a concise answer that directly addresses the user's question. "
                    "Try to provide actual details more than just a summary. "
        )
 
        user_prompt_template = HumanMessagePromptTemplate.from_template(
            "Question: {query}\n\nRetrieved Documents:\n{docs}"
        )
 
        summarization_prompt = ChatPromptTemplate.from_messages(
            [system_prompt_template, user_prompt_template]
        )
 
        formatted_prompt = summarization_prompt.format_messages(query=query,docs=docs)
        response = aws_llm.invoke(formatted_prompt)
        print("\n\n\t--------LLM call DONE for answer summary----------\n\n\tResponse Below:\n\n")
        print(response.content)
        return response.content
 
    except Exception as e:
            print(f"Error during summary generation: {e}")
           
 
def publish_html(run_query_bot_output: List[Dict[str, Any]]) -> Dict[str, str]:
 
    html_content_parts = []
    # Boilerplate HTML and CSS for consistent styling
 
    html_content_parts.append("<!DOCTYPE html>")
 
    html_content_parts.append("<html lang='en'>")
 
    html_content_parts.append("<head>")
 
    html_content_parts.append("<meta charset='UTF-8'>")
 
    html_content_parts.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
 
    html_content_parts.append("<title>Query Bot Results</title>")
 
    html_content_parts.append("<style>")
 
    html_content_parts.append("""
 
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; line-height: 1.6; }
 
        .container { max-width: 900px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
 
        .summary-output {
 
            background-color: #e6f7ff;
 
            border-left: 5px solid #007bff;
 
            padding: 20px;
 
            margin-top: 30px;
 
            border-radius: 5px;
 
            font-family: Calibri, sans-serif;
 
            font-size: 15px;
 
        }
 
        .summary-output h2 { color: #007bff; margin-top: 0; margin-bottom: 15px; text-align: center; font-size: 20px; }
 
    """)
 
    html_content_parts.append("</style>")
 
    html_content_parts.append("</head><body><div class='container'>")
    summary_output_content = None
    # The loop is simplified to find the summary output
 
    for item in run_query_bot_output:
 
        if "output" in item:
 
            summary_output_content = item["output"]
 
            break # Exit loop once the summary is found
    # Add the summary if it exists
 
    if summary_output_content:
 
        html_content_parts.append("<div class='summary-output'><h2>SUMMARY</h2>")
 
        for para in summary_output_content.strip().split('\n\n'):
 
            # Convert markdown to HTML for formatting
 
            converted_html_block = markdown.markdown(para.strip())
 
            html_content_parts.append(converted_html_block)
 
        html_content_parts.append("</div>")
    # Final closing tags
 
    html_content_parts.append("</div></body></html>")
 
    final_html_string = "\n".join(html_content_parts)
    return {"type": "HTML", "Content": final_html_string}
 
app = FastAPI()
@app.websocket("/query")
async def handel_ws(websocket : WebSocket):
    await websocket.accept()
    try:
        while True:
            file_name="qc"
            #---PG conn details------
            host = "localhost"
            database = "postgres"
            user = "Enter-Creds"
            password = "Enter-Creds"
            port = 5432  # Default PostgreSQL port
            connection = connect_to_postgres(host, database, user, password, port)
            query = await websocket.receive_text()
            print(f"Received query: {query}")                                                           
            if connection:
                #file_name=os.path.splitext(os.path.basename(pdf_file))[0]
                #cur=ingest_into_pgvector(connection,chunks,embeddings,file_name)
                print("\n\n\t------PG Connection success--------\n\n")
                docs=search_query(connection,file_name,query)
                output_str=get_answer_from_llm(query,docs)
                output_list=[{"output":output_str}]
                connection.close()
            else:
                output_list=[{"output":"Connection failed"}]
 
            html_result = publish_html(output_list)
            print("Sending bot output to client...")
 
            await websocket.send_text(json.dumps(html_result))
 
    except WebSocketDisconnect:
 
        print("Client Disconnected!!!")
 
 
if __name__ == "__main__":
     uvicorn.run(app , host = "0.0.0.0" , port = 9612)
    
#    #file_name=os.path.splitext(os.path.basename(pdf_file))[0]
#    #file_name="handbook"
#    file_name="qc"
#    #---PG conn details------
#    host = "localhost"
#    database = "postgres"
#    user = "postgres"
#    password = "postgres"
#    port = 5432  # Default PostgreSQL port
#    connection = connect_to_postgres(host, database, user, password, port)                                                                         
#    if connection:
#        #file_name=os.path.splitext(os.path.basename(pdf_file))[0]
#        #cur=ingest_into_pgvector(connection,chunks,embeddings,file_name)
#        docs=search_query(connection,file_name,"Grover algorithm")
#        output_str=get_answer_from_llm(query,docs)
#        output_list=[{"output":output_str}]
#        connection.close()
#        #search_query(connection,file_name,"How does AI affect Customer?")                                       
   
#---------------Extract PDF---------------------------
#pdf_dir=os.path.abspath("pdf_dir")
#
#for root, _, files in os.walk(pdf_dir):
#    for file in files:
#        if file.endswith(".pdf"):
#            pdf_file = os.path.join(root, file)
#            print(f"\nPassing file: {file} for chunk extraction....\n\n")
#            extracted_data = extract_content_page_by_page(pdf_file)
#       
#            if extracted_data:
#                #-----------Extract PDF data save to JSON-------------
#                json_output_path = os.path.join(
#                    "extracted_content_" + os.path.splitext(os.path.basename(pdf_file))[0],
#                    "all_content.json"
#                )
#                with open(json_output_path, 'w') as f:
#                    json.dump(extracted_data, f, indent=4)
#                print(f"Extraction success: {file} saved to '{json_output_path}'.")
#
#                #-------------Load Chunks from JSON----------------
#                print(f"\nGetting saved chunks for file: {file}....\n\n")
#                chunks = get_chunks_from_json(json_output_path)
#                if chunks is not None:
#                    print(f"\nSuccess\n\nGetting Embeddings for file: {file}....\n\n")
#                    embeddings=get_embedding(chunks,"chunks")
#                    print(type(embeddings))
#                    if embeddings is not None:
#                        print(f"\nSuccess\n\n Ingesting file: {file}....\n\n")
#                        connection = connect_to_postgres(host, database, user, password, port)
#                        if connection:
#                              file_name=os.path.splitext(os.path.basename(pdf_file))[0]
#                              #cur=ingest_into_pgvector(connection,chunks,embeddings,file_name)
#                              search_query(connection,file_name,"Grover algorithm")
#                              connection.close()
#                              #search_query(connection,file_name,"How does AI affect Customer?")                                                