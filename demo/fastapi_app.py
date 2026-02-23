from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
import numpy as np
import io
import os

app = FastAPI()

# Load model and processor
model_path = os.environ.get("JANUS_MODEL_PATH", "deepseek-ai/Janus-Pro-1B")
local_files_only = os.environ.get("JANUS_LOCAL_FILES_ONLY", "0") == "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_cuda = device.type == "cuda"
model_dtype = torch.bfloat16 if is_cuda else torch.float32
t2i_parallel_size = int(os.environ.get("JANUS_T2I_PARALLEL_SIZE", "2"))

config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
language_config = config.language_config
language_config._attn_implementation = "sdpa"

if is_cuda:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)

vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                              language_config=language_config,
                                              trust_remote_code=True,
                                              torch_dtype=model_dtype,
                                              low_cpu_mem_usage=True,
                                              local_files_only=local_files_only)
vl_gpt = vl_gpt.to(device=device, dtype=model_dtype).eval()

try:
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path, use_fast=False, local_files_only=local_files_only)
except TypeError:
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path, local_files_only=local_files_only)
tokenizer = vl_chat_processor.tokenizer
cuda_device = device


@torch.inference_mode()
def multimodal_understanding(image_data, question, seed, top_p, temperature):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if is_cuda:
            torch.cuda.manual_seed(seed)

    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{question}",
            "images": [image_data],
        },
        {"role": "Assistant", "content": ""},
    ]

    pil_images = [Image.open(io.BytesIO(image_data))]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=model_dtype if is_cuda else torch.float32)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    do_sample = temperature > 0
    generation_kwargs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": prepare_inputs.attention_mask,
        "pad_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128,
        "do_sample": do_sample,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    outputs = vl_gpt.language_model.generate(**generation_kwargs)
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer


@app.post("/understand_image_and_question/")
async def understand_image_and_question(
    file: UploadFile = File(...),
    question: str = Form(...),
    seed: int = Form(42),
    top_p: float = Form(0.95),
    temperature: float = Form(0.1)
):
    image_data = await file.read()
    response = multimodal_understanding(image_data, question, seed, top_p, temperature)
    return JSONResponse({"response": response})


def generate(input_ids,
             width,
             height,
             temperature: float = 1,
             parallel_size: int = 5,
             cfg_weight: float = 5,
             image_token_num_per_image: int = 576,
             patch_size: int = 16):
    tokens = input_ids.unsqueeze(0).repeat(parallel_size * 2, 1).to(cuda_device, non_blocking=True)
    tokens[1::2, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.empty((parallel_size, image_token_num_per_image), dtype=torch.int, device=cuda_device)

    pkv = None
    for i in range(image_token_num_per_image):
        outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv)
        pkv = outputs.past_key_values
        hidden_states = outputs.last_hidden_state
        logits = vl_gpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)
    patches = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int), 
        shape=[parallel_size, 8, width // patch_size, height // patch_size]
    )

    return generated_tokens.to(dtype=torch.int), patches


def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    return dec.astype(np.uint8)


@torch.inference_mode()
def generate_image(prompt, seed, guidance):
    seed = seed if seed is not None else 12345
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    width = 384
    height = 384
    parallel_size = t2i_parallel_size
    
    with torch.no_grad():
        messages = [{'role': 'User', 'content': prompt}, {'role': 'Assistant', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=''
        )
        text = text + vl_chat_processor.image_start_tag
        input_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=cuda_device)
        _, patches = generate(input_ids, width // 16 * 16, height // 16 * 16, cfg_weight=guidance, parallel_size=parallel_size)
        images = unpack(patches, width // 16 * 16, height // 16 * 16)

        return [Image.fromarray(images[i]).resize((1024, 1024), Image.LANCZOS) for i in range(parallel_size)]


@app.post("/generate_images/")
async def generate_images(
    prompt: str = Form(...),
    seed: int = Form(None),
    guidance: float = Form(5.0),
):
    try:
        images = generate_image(prompt, seed, guidance)
        def image_stream():
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)
                yield buf.read()

        return StreamingResponse(image_stream(), media_type="multipart/related")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("JANUS_FASTAPI_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
