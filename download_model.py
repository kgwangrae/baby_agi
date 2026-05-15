from mlx_vlm import load

print("Downloading Qwen2-VL-2B-Instruct-4bit (approx. 2GB) from HuggingFace")
model, processor = load("mlx-community/Qwen2-VL-2B-Instruct-4bit")
print("Downloaded and compiled the model")

