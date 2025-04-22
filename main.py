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

        prompt = f"{sys_msg}\n\nUser: {user_msg}\nAssistant:"
        print("Manual prompt for model:", prompt)

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
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print("generate_response: Decoded raw response:", response)
        if response.strip() == "<|endoftext|>":
            print("WARNING: Model outputed only <|endoftext|> EOS token. Prompt sent was:", prompt)
        print(f"Generated response: {response}")
        return response

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
        yield error_msg
        return

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
                    max_new_tokens=4096,
                    temperature=0.8,
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
                token_text = tokenizer.decode(new_token)
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
