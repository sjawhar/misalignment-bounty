from dotenv import load_dotenv
from inspect_ai import eval
from eval import bounty
load_dotenv()

MODELS = [
    "anthropic/claude-3-5-sonnet-latest",
    "openai/gpt-4o",
    # "google/gemini-1.5-pro"
]

for model in MODELS:
    print(f"Running model: {model}")
    eval(bounty(), model=model)