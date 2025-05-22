from dotenv import load_dotenv
from inspect_ai import eval_set
from eval import bounty
from datetime import datetime
import os
import shutil
load_dotenv()

MODELS = [
    "anthropic/claude-3-5-sonnet-latest",
    "openai/gpt-4o",
    # "google/gemini-1.5-pro"
]
# Create a folder for logs if it doesn't exist
log_dir = "logs-temp"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
eval_set(bounty(), model=[
    "anthropic/claude-3-5-sonnet-latest",
    "openai/gpt-4o",
    # "google/gemini-1.5-pro"
], log_dir=log_dir)


now = datetime.now().strftime("%Y%m%d_%H%M%S")
destination_dir = f"logs/{now}"
os.makedirs(os.path.dirname(destination_dir), exist_ok=True)
shutil.move(log_dir, destination_dir)