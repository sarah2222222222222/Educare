from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_together import ChatTogether
from pydantic import BaseModel
from typing import Dict, Optional
from PyPDF2 import PdfReader
import io
import uuid

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Initialize Together AI LLM
llm = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0.7,
    max_tokens=1000,
    timeout=None,
    max_retries=2,
    api_key="53f8339c9fae27f01999d39bc7573a39e48ee33ef20f9055dd5599c1838d1f81"  # Replace with your real API key
)

# -------- In-memory PDF text store --------
pdf_store: Dict[str, str] = {}

# -------- Prompt templates --------
SUMMARIZE_PROMPT = """
You are an expert study assistant. Summarize the following text from a PDF in 200-300 words, focusing on key concepts, main ideas, and essential details. Ensure the summary is concise, clear, and suitable for studying. Avoid including unnecessary details.

Text: {pdf_text}
"""

ASSIGNMENT_PROMPT = """
You are an expert study assistant. Based on the following text from a PDF, create a set of study assignments, such as 3-5 questions, exercises, or tasks. The assignments should test understanding of key concepts and encourage critical thinking. Format the output clearly with numbered questions or tasks.

Text: {pdf_text}
"""

MATERIAL_PROMPT = """
You are an expert study assistant. Based on the following user description, generate study material such as concise notes, bullet points, or flashcards. The material should be clear, concise, and optimized for studying, helping the user understand and memorize key concepts effectively. Format the output clearly with headings or numbered items as appropriate.

User Description: {description}
"""

# -------- Helper to extract text from PDF --------
def extract_pdf_text(file: UploadFile) -> str:
    try:
        pdf_reader = PdfReader(io.BytesIO(file.file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

# -------- Upload PDF endpoint --------
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_id = str(uuid.uuid4())  # unique id
    pdf_text = extract_pdf_text(file)
    
    if not pdf_text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
    
    pdf_store[file_id] = pdf_text
    return {"file_id": file_id, "message": "PDF uploaded successfully"}

# -------- Summarize PDF endpoint --------
@app.post("/summarize")
async def summarize(file_id: str):
    if file_id not in pdf_store:
        raise HTTPException(status_code=404, detail="PDF not found")
    pdf_text = pdf_store[file_id]
    prompt = ChatPromptTemplate.from_template(SUMMARIZE_PROMPT)
    chain = prompt | llm
    try:
        result = chain.invoke({"pdf_text": pdf_text}).content
        return {"file_id": file_id, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

# -------- Generate assignments endpoint --------
@app.post("/generate-assignments")
async def generate_assignments(file_id: str):
    if file_id not in pdf_store:
        raise HTTPException(status_code=404, detail="PDF not found")
    pdf_text = pdf_store[file_id]
    prompt = ChatPromptTemplate.from_template(ASSIGNMENT_PROMPT)
    chain = prompt | llm
    try:
        result = chain.invoke({"pdf_text": pdf_text}).content
        return {"file_id": file_id, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating assignments: {str(e)}")

# -------- Study material generation --------
class MaterialRequest(BaseModel):
    description: str

@app.post("/generate-material")
async def generate_material(request: MaterialRequest):
    description = request.description.strip()
    if not description:
        raise HTTPException(status_code=400, detail="Description cannot be empty")
    
    prompt = ChatPromptTemplate.from_template(MATERIAL_PROMPT)
    chain = prompt | llm
    try:
        material = chain.invoke({"description": description}).content
        return {"description": description, "material": material}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating study material: {str(e)}")

# -------- Chat with session history --------

# Load system prompt from file (adjust path if needed)
def get_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

system_prompt = get_system_prompt("assest/system_prompt.txt")

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

runnable: Runnable = prompt | llm
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

class ChatRequest(BaseModel):
    input: str
    session_id: Optional[str] = None

@app.post("/chat")
async def chat(request: ChatRequest = Body(...)):
    session_id = request.session_id or str(uuid.uuid4())
    user_input = request.input

    try:
        result = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return {
            "session_id": session_id,
            "response": result.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chat: {str(e)}")

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    if session_id not in store:
        return JSONResponse(status_code=404, content={"error": "Session not found."})
    messages = store[session_id].messages
    history = [{"type": m.type, "content": m.content} for m in messages]
    return {"session_id": session_id, "history": history}
