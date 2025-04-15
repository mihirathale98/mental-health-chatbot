import pandas as pd
import evaluate

def main():
    # 1. Load the CSV file
    # Replace 'your_file.csv' with the path to your CSV file.
    df = pd.read_csv(r"/content/AIHCI_test.csv")
    
    # Assume the CSV has these columns:
    predictions = df['model_output'].tolist()
    references = df['original_output'].tolist()
    
    # 2. Compute BERTScore
    bertscore_metric = evaluate.load("bertscore")
    bertscore_results = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        lang="en"
    )
    
    # Calculate average F1 score from BERTScore
    avg_bert_f1 = sum(bertscore_results['f1']) / len(bertscore_results['f1'])
    print("Average BERTScore F1:", avg_bert_f1)
    
    # 3. Compute BLEURT
    bleurt_metric = evaluate.load("bleurt")
    bleurt_results = bleurt_metric.compute(
        predictions=predictions,
        references=references
    )
    
    # Calculate average BLEURT score
    avg_bleurt = sum(bleurt_results["scores"]) / len(bleurt_results["scores"])
    print("Average BLEURT Score:", avg_bleurt)
    
if __name__ == '__main__':
    main()
