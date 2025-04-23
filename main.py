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
    # Ensure there is a system prompt
    has_system = any(msg["role"] == "system" for msg in messages)
    if not has_system:
        messages = [{
            "role": "system",
            "content": "Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking processes. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion."
        }] + messages

    # Model or tokenizer failed to load
    if model is None or tokenizer is None:
        error_msg = "[Startup Error] Model or tokenizer was not loaded at application startup. Check container logs for details (common causes: missing HUGGINGFACE_TOKEN, GPU unavailable, model download error)."
        print(error_msg)
        return error_msg

    try:
        print("generate_response: Model name:", getattr(model, 'name_or_path', 'N/A'))
        print("generate_response: Tokenizer name:", getattr(tokenizer, 'name_or_path', 'N/A'))

        # Use chat template (recommended by user)
        chat_input = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        print(f"Input tensor shape: {chat_input.shape}")

        with torch.inference_mode():
            outputs = model.generate(
                chat_input,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        print("generate_response: model.generate() finished.")

        # Decode entire assistant output, including special tokens/formatting
        assistant_output = tokenizer.decode(
            outputs[0][chat_input.shape[-1]:], skip_special_tokens=False
        )
        print("generate_response: Assistant output (raw):", assistant_output)

        if assistant_output.strip() == "<|endoftext|>":
            print("WARNING: Model outputed only <|endoftext|> EOS token. Raw response was:", assistant_output)
        print(f"Generated response: {assistant_output}")
        return assistant_output

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
            "content": "Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking processes. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion."
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

        # Generate with streamer - Reverted buffer logic
        # buffered_output = ""
        # in_think_block = True 

        # Generate with proper error handling
        # print("Stream: Entering token generation loop...") # Removed log
        with torch.inference_mode():
            streamer_output_debug = "" # Add simple debug accumulator
            for i in range(max_new_tokens):
                # print(f"Stream: Starting generation for token {i+1}...") # Removed log
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
                # print(f"Stream: model.generate completed for token {i+1}") # Removed log

                # Get the newly generated token ID
                new_token_id = outputs[0][-1].item() # Get the last token ID

                # If we hit the end token, stop
                if new_token_id == tokenizer.eos_token_id:
                    break

                # Decode the single token, DO skip special tokens for cleaner output now
                token_text = tokenizer.decode([new_token_id], skip_special_tokens=True) 
                # print(f"Stream: Raw token decoded: '{token_text}'") # Removed log

                # Update input_ids for next token generation by appending the new token
                input_ids = torch.cat([input_ids, outputs[:, -1:]], dim=-1)
                if attention_mask is not None:
                     attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
                
                # Reverted: Removed buffering logic
                # print(f"Stream: Current state - in_think_block={in_think_block}") 
                # if in_think_block:
                #    ... buffer logic ...
                # else:
                
                # Yield the token text directly
                if token_text: # Avoid yielding empty strings if skip_special_tokens removes everything
                    streamer_output_debug += token_text # Accumulate for debug
                    # print(f"Stream: Yielding subsequent token: '{token_text}'") # Removed log
                    yield token_text

                # Small delay to control stream rate
                await asyncio.sleep(0.01)
        
        # Reverted: Removed end-of-stream check for in_think_block
        # if in_think_block:
        #     print("Warning: Stream ended while still processing the <think> block.")
            
        # Debug output of accumulated stream
        print(f"Generated streaming response complete (debug): {streamer_output_debug}") 

    except Exception as e:
        print(f"Error in generate_stream: {str(e)}")
        yield f"[Runtime Error] {str(e)}"

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    full_output = generate_response(request.messages, request.max_new_tokens)
    # Extract only the Solution part (after </think>)
    solution = ""
    if full_output:
        think_tag = "</think>"
        idx = full_output.find(think_tag)
        if idx != -1:
            solution = full_output[idx + len(think_tag):].strip()
        else:
            solution = full_output.strip()  # fallback if structure is broken
    return {
        "full_output": full_output,
        "solution": solution,
    }

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    return StreamingResponse(
        generate_stream(request.messages, request.max_new_tokens),
        media_type="text/plain"
    )
