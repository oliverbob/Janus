import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image

import numpy as np
import os


# Load model and processor
model_path = os.environ.get("JANUS_MODEL_PATH", "deepseek-ai/Janus-1.3B")
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
# Multimodal Understanding function
@torch.inference_mode()
# Multimodal Understanding function
def multimodal_understanding(image, question, seed, top_p, temperature):
    if image is None:
        return "Please upload an image first."
    if not question:
        return "Please enter a question."
    
    # set seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if is_cuda:
            torch.cuda.manual_seed(seed)
    
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "Assistant", "content": ""},
    ]
    
    pil_images = [Image.fromarray(image)]
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
        outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds,
                                             use_cache=True,
                                             past_key_values=pkv)
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
    patches = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                 shape=[parallel_size, 8, width // patch_size, height // patch_size])

    return generated_tokens.to(dtype=torch.int), patches

def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    return dec.astype(np.uint8)



@torch.inference_mode()
def generate_image(prompt,
                   seed=None,
                   guidance=5):
    # Set the seed for reproducible results
    if seed is not None:
        torch.manual_seed(seed)
        if is_cuda:
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    width = 384
    height = 384
    parallel_size = t2i_parallel_size
    
    with torch.no_grad():
        messages = [{'role': 'User', 'content': prompt},
                    {'role': 'Assistant', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=messages,
                                                                   sft_format=vl_chat_processor.sft_format,
                                                                   system_prompt='')
        text = text + vl_chat_processor.image_start_tag
        input_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=cuda_device)
        output, patches = generate(input_ids,
                                   width // 16 * 16,
                                   height // 16 * 16,
                                   cfg_weight=guidance,
                                   parallel_size=parallel_size)
        images = unpack(patches,
                        width // 16 * 16,
                        height // 16 * 16)

        return [Image.fromarray(images[i]).resize((1024, 1024), Image.LANCZOS) for i in range(parallel_size)]

        

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(value="# Multimodal Understanding")
    # with gr.Row():
    with gr.Row():
        image_input = gr.Image()
        with gr.Column():
            question_input = gr.Textbox(label="Question")
            und_seed_input = gr.Number(label="Seed", precision=0, value=42)
            top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="top_p")
            temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="temperature")
        
    understanding_button = gr.Button("Chat")
    understanding_output = gr.Textbox(label="Response")

    examples_inpainting = gr.Examples(
        label="Multimodal Understanding examples",
        examples=[
            [
                "explain this meme",
                "images/doge.png",
            ],
            [
                "Convert the formula into latex code.",
                "images/equation.png",
            ],
        ],
        inputs=[question_input, image_input],
    )
    
        
    gr.Markdown(value="# Text-to-Image Generation")

    
    
    with gr.Row():
        cfg_weight_input = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight")

    prompt_input = gr.Textbox(label="Prompt")
    seed_input = gr.Number(label="Seed (Optional)", precision=0, value=12345)

    generation_button = gr.Button("Generate Images")

    image_output = gr.Gallery(label="Generated Images", columns=2, rows=2, height=300)

    examples_t2i = gr.Examples(
        label="Text to image generation examples. (Tips for designing prompts: Adding description like 'digital art' at the end of the prompt or writing the prompt in more detail can help produce better images!)",
        examples=[
            "Master shifu racoon wearing drip attire as a street gangster.",
            "A cute and adorable baby fox with big brown eyes, autumn leaves in the background enchanting,immortal,fluffy, shiny mane,Petals,fairyism,unreal engine 5 and Octane Render,highly detailed, photorealistic, cinematic, natural colors.",
            "The image features an intricately designed eye set against a circular backdrop adorned with ornate swirl patterns that evoke both realism and surrealism. At the center of attention is a strikingly vivid blue iris surrounded by delicate veins radiating outward from the pupil to create depth and intensity. The eyelashes are long and dark, casting subtle shadows on the skin around them which appears smooth yet slightly textured as if aged or weathered over time.\n\nAbove the eye, there's a stone-like structure resembling part of classical architecture, adding layers of mystery and timeless elegance to the composition. This architectural element contrasts sharply but harmoniously with the organic curves surrounding it. Below the eye lies another decorative motif reminiscent of baroque artistry, further enhancing the overall sense of eternity encapsulated within each meticulously crafted detail. \n\nOverall, the atmosphere exudes a mysterious aura intertwined seamlessly with elements suggesting timelessness, achieved through the juxtaposition of realistic textures and surreal artistic flourishes. Each component\u2014from the intricate designs framing the eye to the ancient-looking stone piece above\u2014contributes uniquely towards creating a visually captivating tableau imbued with enigmatic allure.",
        ],
        inputs=prompt_input,
    )
    
    understanding_button.click(
        multimodal_understanding,
        inputs=[image_input, question_input, und_seed_input, top_p, temperature],
        outputs=understanding_output
    )
    
    generation_button.click(
        fn=generate_image,
        inputs=[prompt_input, seed_input, cfg_weight_input],
        outputs=image_output
    )

try:
    demo.queue(concurrency_count=1, max_size=8).launch(share=False)
except TypeError:
    demo.queue(default_concurrency_limit=1, max_size=8).launch(share=False)