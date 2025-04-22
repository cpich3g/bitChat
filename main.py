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

model_id = "microsoft/phi-4-reasoning"
model = None
tokenizer = None
device = None

import subprocess

try:
    print("=== PACKAGE DIAGNOSTICS START ===")
    try:
        import accelerate
        print("accelerate is importable!")
    except ImportError:
        print("accelerate is NOT importable!")

    print("--- pip list output ---")
    pip_list = subprocess.run(["pip", "list"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")
    print(pip_list.stdout)
    print("--- END pip list output ---")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available! Model load will fail if CUDA is required.")
    print("=== PACKAGE DIAGNOSTICS END ===")

    device = torch.device("cuda")
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token is None:
        raise RuntimeError("HUGGINGFACE_TOKEN environment variable is not set. Model loading will fail.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        token=token,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

    # Optional: torch.compile for even faster inference (PyTorch 2.x+)
    try:
        model = torch.compile(model)
    except Exception:
        pass
    print("Model and tokenizer loaded successfully.")

except Exception as e:
    print(f"Error loading model/tokenizer: {e}")


def generate_response(messages: List[Dict[str, str]], max_new_tokens: int = 4096) -> str:
    # Create a system message if one doesn't exist
    has_system = any(msg["role"] == "system" for msg in messages)
    if not has_system:
        messages = [{
            "role": "system",
            "content": "Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking processes. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion."
        }] + messages

    # Model or tokenizer failed to load
    if model is None or tokenizer is None:
        error_msg = "[Startup Error] Model or tokenizer was not loaded at application startup. Check container logs for details (common causes: missing HUGGINGFACE_TOKEN, GPU unavailable, model download error)."
        print(error_msg)
        return error_msg

    try:
        # First tokenize without moving to device
        print("generate_response: Model name:", getattr(model, 'name_or_path', 'N/A'))
        print("generate_response: Tokenizer name:", getattr(tokenizer, 'name_or_path', 'N/A'))

        # Manual prompt construction (no chat template)
        sys_msg = ""
        user_msg = ""
        for m in messages:
            if m["role"] == "system":
                sys_msg = m["content"]
            elif m["role"] == "user":
                user_msg = m["content"]

        # Phi-4 and similar models require a chat template with special tokens.
        prompt = (
            "<|im_start|>system\n"
            f"{sys_msg.strip()}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_msg.strip()}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        print("Manual prompt for PHI model:", prompt)

        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device) if hasattr(enc, "attention_mask") else None

        print(f"Input tensor shape: {input_ids.shape}")

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        print("generate_response: model.generate() finished.")

        # Decode only newly generated part
        response_raw = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print("generate_response: Decoded raw response:", response_raw)
        
        # Remove the <think> block
        import re
        response_processed = re.sub(r"<think>.*?</think>\s*", "", response_raw, flags=re.DOTALL).strip()
        
        if response_processed.strip() == "<|endoftext|>":
             print("WARNING: Model outputed only <|endoftext|> EOS token after processing. Raw response was:", response_raw)
        print(f"Processed response (no think block): {response_processed}")
        return response_processed

    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        return f"[Runtime Error] {str(e)}"

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_new_tokens: int = 4096

async def generate_stream(messages: List[Dict[str, str]], max_new_tokens: int = 4096) -> AsyncIterable[str]:
    # Create a system message if one doesn't exist
    has_system = any(msg["role"] == "system" for msg in messages)
    if not has_system:
        messages = [{
            "role": "system",
            "content": "Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking processes. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion."
        }] + messages

    # Model or tokenizer failed to load
    if model is None or tokenizer is None:
        error_msg = "[Startup Error] Model or tokenizer was not loaded at application startup. Check container logs for details (common causes: missing HUGGINGFACE_TOKEN, GPU unavailable, model download error)."
        print(error_msg)
        print("Stream: Yielding model/tokenizer load error...") # Log before yielding error
        yield error_msg
        return

    try:
        # Manual prompt construction for Phi-4 format
        sys_msg = ""
        user_msg = ""
        for m in messages:
            if m["role"] == "system":
                sys_msg = m["content"]
            elif m["role"] == "user":
                user_msg = m["content"]

        prompt = (
            "<|im_start|>system\n"
            f"{sys_msg.strip()}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_msg.strip()}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        print("Manual prompt for PHI model (stream):", prompt)

        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device) if hasattr(enc, "attention_mask") else None

        print(f"Input tensor shape (stream): {input_ids.shape}")

        # Generate with streamer
        buffered_output = ""
        in_think_block = True # Assume response starts within <think> as per prompt

        # Generate with proper error handling
        print("Stream: Entering token generation loop...") # Log before loop starts
        with torch.inference_mode():
            for i in range(max_new_tokens):
                print(f"Stream: Starting generation for token {i+1}...") # Log start of iteration
                # Generate one token at a time
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1, # Generate one token at a time for streaming
                    temperature=0.8,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                print(f"Stream: model.generate completed for token {i+1}") # Log after generate call

                # Get the newly generated token ID
                new_token_id = outputs[0][-1].item() # Get the last token ID

                # If we hit the end token, stop
                if new_token_id == tokenizer.eos_token_id:
                    break

                # Decode the single token, DO NOT skip special tokens initially for processing
                token_text = tokenizer.decode([new_token_id], skip_special_tokens=False)
                print(f"Stream: Raw token decoded: '{token_text}'") # Log raw token

                # Update input_ids for next token generation by appending the new token
                input_ids = torch.cat([input_ids, outputs[:, -1:]], dim=-1)
                if attention_mask is not None:
                     attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
                
                print(f"Stream: Current state - in_think_block={in_think_block}") # Log state
                if in_think_block:
                    buffered_output += token_text
                    print(f"Stream: Appended to buffer. Buffer size: {len(buffered_output)}") # Log buffer append
                    # Check if the end tag is now in the buffer
                    end_tag_index = buffered_output.find("</think>")
                    if end_tag_index != -1:
                        print("Stream: Found '</think>' tag in buffer.") # Log tag found
                        # Found the end tag. Extract content after it.
                        start_yielding_index = end_tag_index + len("</think>")
                        content_to_yield = buffered_output[start_yielding_index:].lstrip() # Remove leading space after tag
                        
                        # Decode the part to yield again, this time skipping special tokens for cleaner output
                        # This is tricky as we only have text now. Let's just yield the processed text.
                        # Re-tokenizing and decoding might be complex here.
                        # We will rely on the frontend markdown renderer to handle remaining special tokens if any.
                        
                        if content_to_yield:
                             processed_yield = content_to_yield.replace("<0x0A>", "\n")
                             print(f"Stream: Yielding content after think block: '{processed_yield}'") # Log yield
                             yield processed_yield
                        
                        in_think_block = False
                        print("Stream: Set in_think_block = False") # Log state change
                        buffered_output = "" # Clear buffer
                        print("Stream: Cleared buffer.") # Log buffer clear
                else:
                    # Think block is finished, yield subsequent tokens directly
                    processed_yield = token_text.replace("<0x0A>", "\n")
                    print(f"Stream: Yielding subsequent token: '{processed_yield}'") # Log subsequent yield
                    yield processed_yield

                # Small delay to control stream rate
                await asyncio.sleep(0.01)
        
        # If loop finished but still in think block (e.g., model stopped early), yield nothing more.
        if in_think_block:
            print("Warning: Stream ended while still processing the <think> block.")
            
        # Debug output - streamer_output is no longer tracked this way
        # print(f"Generated streaming response complete: {streamer_output}") 

    except Exception as e:
        print(f"Error in generate_stream: {str(e)}")
        yield f"[Runtime Error] {str(e)}"

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
