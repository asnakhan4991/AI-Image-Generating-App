import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

# API Keys
HF_TOKEN = "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"  # Replace with your Hugging Face token
MODEL_ID = "runwayml/stable-diffusion-v1-5"


def load_pipeline():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_auth_token=HF_TOKEN
    ).to(device)

    
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xFormers enabled")
    except Exception as e:
        print(f"⚠️ xFormers not available: {e}")

    return pipe

pipe = load_pipeline()

# To Generate Image
def generate_image(prompt, steps, guidance):
    image = pipe(prompt, guidance_scale=guidance, num_inference_steps=steps).images[0]
    return image

# Gradio
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", value="A futuristic cityscape at sunset"),
        gr.Slider(10, 50, value=25, step=1, label="Inference Steps"),
        gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
    ],
    outputs=gr.Image(type="pil"),
    title="🚀 AI Image Generator",
    description="Generate images using Stable Diffusion (Gradio + Diffusers)."
)

demo.launch()
