import os
import json
import lm_eval
from lm_eval.models.huggingface import HFLM

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Your models
BASE_MODEL_ID = "google/gemma-4-E2B-it"
UNCENSORED_MODEL_ID = "Kasper-Bankler/gemma-4-E2B-uncensored"

# The benchmarks we want to run
TASKS = ["arc_easy", "hellaswag", "truthfulqa_mc2"]

OUTPUT_FILE = "benchmark_results.json"

# ==========================================
# 2. EVALUATION FUNCTION
# ==========================================
def evaluate_model(model_path, model_name):
    print("\n" + "="*80)
    print(f"🚀 STARTING BENCHMARK: {model_name}")
    print("="*80)
    
    # Initialize the model using lm-eval's special Hugging Face wrapper
    model_wrapper = HFLM(
        pretrained=model_path,
        device="cuda",
        dtype="bfloat16",
        batch_size="auto"
    )

    # Run the evaluation
    print(f"Running tasks: {TASKS}...")
    results = lm_eval.simple_evaluate(
        model=model_wrapper,
        tasks=TASKS,
        num_fewshot=0, 
        log_samples=False,
        limit=500 # Remove this to evaluate on the full dataset
    )
    
    # Extract just the core accuracy metrics
    metrics = {}
    for task_name, task_results in results['results'].items():
        # Most tasks use 'acc,none' for accuracy, truthfulqa uses 'acc,none' or 'mc2'
        accuracy = task_results.get('acc,none', task_results.get('acc', 0.0))
        metrics[task_name] = round(accuracy * 100, 2)
        print(f"  -> {task_name.upper()} Score: {metrics[task_name]}%")

    # Free up VRAM for the next model
    del model_wrapper
    import torch
    torch.cuda.empty_cache()
    
    return metrics

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    final_results = {}

    # 1. Benchmark the Normal Model
    base_metrics = evaluate_model(BASE_MODEL_ID, "Normal Gemma 4")
    final_results["Normal_Gemma"] = base_metrics

    # 2. Benchmark the Uncensored Model
    uncensored_metrics = evaluate_model(UNCENSORED_MODEL_ID, "Uncensored Gemma 4")
    final_results["Uncensored_Gemma"] = uncensored_metrics

    # 3. Save to disk so you can put it in your paper
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_results, f, indent=4)

    # 4. Print the final comparative table
    print("\n\n" + "="*80)
    print(" 📊 FINAL BENCHMARK COMPARISON")
    print("="*80)
    print(f"{'Task':<20} | {'Normal Gemma':<15} | {'Uncensored Gemma':<15}")
    print("-" * 55)
    
    for task in TASKS:
        base_score = f"{final_results['Normal_Gemma'][task]}%"
        unc_score = f"{final_results['Uncensored_Gemma'][task]}%"
        print(f"{task:<20} | {base_score:<15} | {unc_score:<15}")
        
    print("\nResults saved to 'benchmark_results.json'.")