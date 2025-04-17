from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, AsyncIterable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import os
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

@app.get("/")
def serve_root():
    index_path = os.path.join("static", "index.html")
    return FileResponse(index_path)

model_id = "microsoft/bitnet-b1.58-2B-4T"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
).to(device)
model.eval()

# Optional: torch.compile for even faster inference (PyTorch 2.x+)
try:
    model = torch.compile(model)
except Exception:
    pass

def generate_response(messages: List[Dict[str, str]], max_new_tokens: int = 500) -> str:
    # Create a system message if one doesn't exist
    has_system = any(msg["role"] == "system" for msg in messages)
    if not has_system:
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}] + messages
    
    try:
        # First tokenize without moving to device
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        # Move tensors to device separately
        input_ids = inputs.to(device)
        
        # Generate with proper error handling
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Get only the newly generated tokens
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Debug output
        print(f"Generated response: {response}")
        return response
        
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        return "I'm sorry, I encountered an error while generating a response."

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_new_tokens: int = 500

async def generate_stream(messages: List[Dict[str, str]], max_new_tokens: int = 500) -> AsyncIterable[str]:
    # Create a system message if one doesn't exist
    has_system = any(msg["role"] == "system" for msg in messages)
    if not has_system:
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}] + messages
    
    try:
        # First tokenize without moving to device
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        # Move tensors to device separately
        input_ids = inputs.to(device)
        
        # Generate with streamer
        streamer_output = ""
        
        # Generate with proper error handling
        with torch.inference_mode():
            for i in range(max_new_tokens):
                # Generate one token at a time
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=1,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                # Get the newly generated token
                new_token = outputs[0][input_ids.shape[1]:][0]
                
                # If we hit the end token, stop
                if new_token.item() == tokenizer.eos_token_id:
                    break
                
                # Decode the single token
                token_text = tokenizer.decode(new_token, skip_special_tokens=True)
                streamer_output += token_text
                
                # Update input_ids for next token generation
                input_ids = outputs
                
                # Yield the token text directly, no SSE formatting
                yield token_text
                
                # Small delay to control stream rate
                await asyncio.sleep(0.01)
        
        # Debug output
        print(f"Generated streaming response complete: {streamer_output}")
        
    except Exception as e:
        print(f"Error in generate_stream: {str(e)}")
        yield "I'm sorry, I encountered an error while generating a response."

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    response = generate_response(request.messages, request.max_new_tokens)
    return {"response": response}

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    return StreamingResponse(
        generate_stream(request.messages, request.max_new_tokens),
        media_type="text/plain"
    )
