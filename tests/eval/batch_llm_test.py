import os
from run_generation_eval import run_full_evaluation_with_config


def test_run_full_evaluation(test_models: list, test_text_dict: dict, style="essay"):
    """Test the full evaluation process for grammar, style, and clarity."""
    
    total_runs = len(test_models) * len(test_text_dict)
    counter = 1
    print(f"Starting batch evaluation: {total_runs} runs ({len(test_models)} models x {len(test_text_dict)} texts)\n")

    for model in test_models:        
        print(f"\n{'='*50}")
        print(f"Testing model: {model}")
        print(f"{'='*50}")
        
        # Override BOTH the config dict and environment variable
        # EVAL_CONFIG['llm_model'] = model
        # EVAL_CONFIG["result_file_name"] = "eval_generation_results.jsonl"
        # os.environ['LOCAL_MODEL'] = model

        for key, value in test_text_dict.items():
            print(f"Evaluation {counter}/{total_runs} - Testing text: {key}")
            # EVAL_CONFIG["file_name"] = key
            counter += 1
            
            try:
                # run_full_evaluation(value, style=style)
                run_full_evaluation_with_config(value, style=style, llm_model=model, result_file_name="eval_generation_results.jsonl", file_name=key)
            except Exception as e:
                print(f"ERROR with {model} on {key}: {e}")
                continue  # Continue to next text even if this one fails




def get_essay_dict():
    """Gather all the text files for Essays"""
    essay_folder = "data/eval/texts/test_essay/"
    essay_files = [f for f in os.listdir(essay_folder) if f.endswith(".txt")]
    essay_dict = {}

    for file in essay_files:
        file_path = os.path.join(essay_folder, file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            essay_dict[file] = content

    return essay_dict

def get_fiction_dict():
    """Gather all the text files for Fiction"""
    fiction_folder = "data/eval/texts/test_fiction/"
    fiction_files = [f for f in os.listdir(fiction_folder) if f.endswith(".txt")]
    fiction_dict = {}

    for file in fiction_files:
        file_path = os.path.join(fiction_folder, file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            fiction_dict[file] = content
            
    return fiction_dict

if __name__ == "__main__":

    """This will run the full evaluation process for multiple models and multiple texts in a batch manner and add them to the eval_result.json"""
    test_models = [
        # "qwen3.5:4b", # Gets stuck loading forever. Too small to work with our context
        # "llama3.1",
        # "mistral", # 7b
        # "ministral-3:8b",
        # "minstral-nemo:12b",
        # "qwen2.5:7b",
        # "qwen2.5:14b",
        # "qwen3:8b",
        # "qwen3:14b",
        # "mistral-small3.2:24b",
        "qwen3.5:9b",  # Run everything over this one as well...
        # "qwen3.6:27b", # Ridicoulous example
        # kimi 
        # grok
    ] 

    essays = get_essay_dict()
    # fiction = get_fiction_dict()

    # Remove some items for testing
    # essays.pop("Article_C2_Profishency_Response_3.txt")
    # essays.pop("Wider_Audience_A1.txt")

    
    test_run_full_evaluation(test_models, essays, style="essay")
    #test_run_full_evaluation(test_models, fiction, style="fiction")

        