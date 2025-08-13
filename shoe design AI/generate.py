#generate.py


import os
import torch
from diffusers import StableDiffusionPipeline

# Constants
MODEL_ID = "stabilityai/stable-diffusion-2-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "../generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the model once
print("🔄 Loading Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
pipe.to(DEVICE)
print("✅ Model loaded and ready.")


def generate_image(
    prompt: str,
    filename: str = "output.png",
    steps: int = 60,
    scale: float = 10.0,
    height: int = 768,
    width: int = 768 ,
    negative_prompt: str = "blurry, noisy, low quality, back view, cropped, out of frame, bad composition, more than 2 shoe image, extra limbs, extra legs, duplicate feet, deformed feet, malformed anatomy, disfigured, mutation, long legs, disconnected limbs, poorly drawn, fused shoes, bad anatomy, cloned legs, ugly"
) -> str:
    """Generate an image from a prompt and save it."""
    print(f"🎨 Generating image for prompt: {prompt}")
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=scale,
            height=height,
            width=width
        )
        filepath = os.path.join(OUTPUT_DIR, filename)
        result.images[0].save(filepath)
        print(f"✅ Image saved at: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Error during image generation: {e}")
        return ""


# Example usage
if __name__ == "__main__":
    test_prompt = "A sleek runner-style sneaker with a slim profile, featuring a smooth seamless upper made from breathable mesh in deep berry tones. The shoe has a simple, unadorned design with a white heel counter and minimal branding. Displayed on a sleek modern white background with soft ambient lighting. Hyper-realistic, 8k ultra-detailed render."

    generate_image(test_prompt, filename="shoe_output.png")