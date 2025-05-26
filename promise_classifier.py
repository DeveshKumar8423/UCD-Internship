import os
import subprocess
import pandas as pd
import time
from codecarbon import EmissionsTracker

# 1. Data Loading with Error Handling
def load_promise_data(folder_path):
    """Load all PROMISE dataset files with flexible parsing"""
    files = {
        'train': 'train.txt',
        'train_d': 'traind.txt',
        'test': 'test.txt',
        'test_d': 'testd.txt'
    }
    
    dfs = {}
    for name, filename in files.items():
        try:
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                data = []
                for line in lines:
                    parts = line.split(',')
                    if len(parts) == 3:
                        label = parts[0].strip()
                        text = parts[1].strip()
                    elif len(parts) == 2:
                        label = parts[0].strip()
                        text = parts[1].strip()
                    else:
                        continue  # skip malformed lines
                    data.append([label, text])
                dfs[name] = pd.DataFrame(data, columns=['label', 'text'])
            print(f"Successfully loaded {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            dfs[name] = pd.DataFrame(columns=['label', 'text'])
    return dfs

# 2. Enhanced Local Ollama Interface
def query_ollama(prompt, model="mistral", max_retries=3):
    """Run Ollama locally with retry logic"""
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                check=True,
                timeout=60  
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            print(f"Timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                return "TIMEOUT_ERROR"
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed: {e.stderr}")
            if attempt == max_retries - 1:
                return "PROCESS_ERROR"
        time.sleep(2)  # Wait before retry
    return "MAX_RETRIES_EXCEEDED"

# 3. Robust Classification Function
def classify_requirement(text):
    """Classify requirement with better error handling"""
    prompt = f"""
    Classify this software requirement:
    - Reply ONLY with 'F' for Functional (describes WHAT the system does)
    - Reply ONLY with 'NF' for Non-Functional (describes HOW the system behaves)

    Examples:
    [F] "User login feature"
    [NF] "Response time under 2 seconds"
    [NF] "System must be available 99.9% of the time"

    Requirement: "{text}"
    Classification:"""
    
    response = query_ollama(prompt).upper()
    
    # Enhanced response parsing
    if not response:
        return "EMPTY_RESPONSE"
    elif response.startswith(('F', 'FUNCTIONAL')):
        return 'F'
    elif response.startswith(('NF', 'NON-FUNCTIONAL', 'NONFUNCTIONAL')):
        return 'NF'
    else:
        return f"UNKNOWN_RESPONSE_{response[:20]}"

# Main Execution
if __name__ == "__main__":
    # Load all datasets
    promise_folder = "../../../Downloads/PROMISE "  # Folder containing files
    datasets = load_promise_data(promise_folder)
    
    # Verify having data
    if all(df.empty for df in datasets.values()):
        raise FileNotFoundError("No valid data files found in PROMISE folder")
    
    # Combine train and test sets
    train_df = pd.concat([datasets['train'], datasets['train_d']], ignore_index=True)
    test_df = pd.concat([datasets['test'], datasets['test_d']], ignore_index=True)
    
    # Clean and prepare data
    def clean_data(df):
        df['label'] = df['label'].str.upper().replace(
            {'FR': 'F', 'FUNCTIONAL': 'F', 
             'NFR': 'NF', 'NON-FUNCTIONAL': 'NF', 'NONFUNCTIONAL': 'NF'}
        )
        df = df[df['label'].isin(['F', 'NF'])]  # Filter valid labels
        df['text'] = df['text'].str.strip()
        return df.dropna()
    
    clean_train = clean_data(train_df)
    clean_test = clean_data(test_df)
    
    print(f"\nTraining samples: {len(clean_train)}")
    print(f"Test samples: {len(clean_test)}")
    print("Label distribution in test set:")
    print(clean_test['label'].value_counts())
    
    # Run evaluation
    os.makedirs("emissions", exist_ok=True)  # Ensure emissions folder exists
    tracker = EmissionsTracker(
        project_name="PROMISE_Clasisification",
        measure_power_secs=1,
        output_dir="emissions"
    )
    
    results = []
    tracker.start()

    for _, row in clean_test.iterrows():
        start_time = time.time()
        pred = classify_requirement(row['text'])
        latency = time.time() - start_time

        results.append({
            'text': row['text'][:100] + "...",
            'true_label': row['label'],
            'pred_label': pred,
            'match': row['label'] == pred,
            'latency_sec': latency
            # emissions_kgco2 will be add soon
        })

    emissions = tracker.stop()
    results_df = pd.DataFrame(results)

    # Estimate per-prediction emissions
    if len(results_df) > 0:
        emission_per_pred = emissions / len(results_df)
        results_df['emissions_kgco2'] = emission_per_pred
    else:
        results_df['emissions_kgco2'] = 0.0

    results_df.to_csv("results/classification_results.csv", index=False)
    
    # Analysis
    if results_df.empty:
        print("\nNo predictions were made. Check your test data and classification function.")
    else:
        valid_results = results_df[results_df['pred_label'].isin(['F', 'NF'])]
        
        if not valid_results.empty:
            accuracy = valid_results['match'].mean()
            print("\nClassification Report:")
            print(valid_results[['text', 'true_label', 'pred_label', 'match']])
            print(f"\nAccuracy (valid responses): {accuracy:.1%}")
        else:
            print("\nNo valid classifications obtained")
        
        print("\nAll Responses:")
        print(results_df['pred_label'].value_counts())
        
        print(f"\nEnergy used: {emissions:.6f} kgCO2")
        print(f"Average latency: {results_df['latency_sec'].mean():.2f}s")
        
        # Save full results
        os.makedirs("results", exist_ok=True)
        results_df.to_csv("results/classification_results.csv", index=False)
        print("\nResults saved to results/classification_results.csv")