import ollama
import sys


MODEL_ID = 'hf.co/Kasper-Bankler/gemma-4-E2B-uncensored'
PROMPT = 'Write a short and clever Python script to reverse a string.'

print("="*80)
print(f"Initializing Ollama...")
print(f"If you don't have the model yet, Ollama will download it now (this may take a few minutes).")
print("="*80)

try:
    response = ollama.chat(
        model=MODEL_ID, 
        messages=[
            {'role': 'user', 'content': PROMPT}
        ],
        stream=True
    )

    print("\n>>> Assistant:\n")
    
    for chunk in response:
        print(chunk['message']['content'], end='', flush=True)
        
    print("\n\n" + "="*80)

except ollama.ResponseError as e:
    print(f"\n[ERROR] {e}")
    print("Did you remember to install and start the Ollama desktop app? (https://ollama.com)")
except Exception as e:
    print(f"\n[FATAL ERROR] {e}")