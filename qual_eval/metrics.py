import os
import json
import logging
import pandas as pd
import time
from datasets import Dataset
from ragas.metrics import AspectCritic, RubricsScore
from ragas import evaluate
from constants import EMPATHY_CRITERIA, EMPATHY_RUBRIC, EVASIVENESS_RUBRIC, HELPFULNESS_RUBRIC, SAFETY_RUBRIC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Batch size configuration
BATCH_SIZE = 5  # Process 5 samples at a time
BATCH_DELAY = 2  # Wait 2 seconds between batches to avoid rate limits

def load_test_data(filepath):
    """Load test data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def load_model_responses(filepath):
    """Load model responses from JSON file"""
    with open(filepath, 'r') as f:
        responses = json.load(f)
    
    # Check the structure of the responses to determine how to extract content
    processed_responses = []
    for response in responses:
        if 'choices' in response and response['choices'] and len(response['choices']) > 0:
            if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                processed_responses.append(response['choices'][0]['message']['content'])
            elif 'text' in response['choices'][0]:
                processed_responses.append(response['choices'][0]['text'])
        else:
            # If the expected structure isn't found, add a placeholder
            processed_responses.append("")
    
    return processed_responses

def define_metrics():
    """Create the evaluation metrics"""
    empathy_aspect = AspectCritic(
        name="empathy_check",
        definition=EMPATHY_CRITERIA,
        strictness=1,
        max_retries=5
    )

    # Convert tuple to dictionary with numbered keys if needed
    if isinstance(EMPATHY_RUBRIC, tuple):
        rubrics_dict = {f"rubric_{i}": rubric for i, rubric in enumerate(EMPATHY_RUBRIC)}
        rubric_empathy = RubricsScore(
            name="empathy_score",
            rubrics=rubrics_dict,
        )
    else:
        rubric_empathy = RubricsScore(
            name="empathy_score",
            rubrics=EMPATHY_RUBRIC,
        )
    
    if isinstance(SAFETY_RUBRIC, tuple):
        safety_rubric_dict = {f"rubric_{i}": rubric for i, rubric in enumerate(SAFETY_RUBRIC)}
        rubric_safety = RubricsScore(
            name="safety_score",
            rubrics=safety_rubric_dict,
        )
    else:    
        rubric_safety = RubricsScore(
            name="safety_score",
            rubrics=SAFETY_RUBRIC,
        )
        
    if isinstance(HELPFULNESS_RUBRIC, tuple):
        helpfulness_rubric_dict = {f"rubric_{i}": rubric for i, rubric in enumerate(HELPFULNESS_RUBRIC)}
        helpfulness_rubric = RubricsScore(
            name="helpfulness_score",
            rubrics=helpfulness_rubric_dict,
        )
    else:
        helpfulness_rubric = RubricsScore(
            name="helpfulness_score",
            rubrics=HELPFULNESS_RUBRIC,
        )
    
    if isinstance(EVASIVENESS_RUBRIC, tuple):
        evasiveness_rubric_dict = {f"rubric_{i}": rubric for i, rubric in enumerate(EVASIVENESS_RUBRIC)}
        evasiveness_rubric = RubricsScore(
            name="evasiveness_score",
            rubrics=evasiveness_rubric_dict,
        )
    else:
        evasiveness_rubric = RubricsScore(
            name="evasiveness_score",
            rubrics=EVASIVENESS_RUBRIC,
        )
        
    return [empathy_aspect, rubric_empathy, rubric_safety, helpfulness_rubric, evasiveness_rubric]

def create_dataset_for_evaluation(questions, answers):
    """Create a dataset for RAGAS evaluation"""
    data = []
    for q, a in zip(questions, answers):
        if not a:  # Skip empty responses
            continue
        data.append({
            "question": q,
            "answer": a
        })
    return Dataset.from_list(data)

def batch_generator(data, batch_size):
    """Generate batches from a dataset"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def run_evaluation_in_batches(dataset, metrics, model_name):
    """Run the RAGAS evaluation in batches to avoid rate limits"""
    try:
        all_results = {}
        dataset_dict = dataset.to_dict()
        total_samples = len(dataset)
        dataset_list = [{"question": dataset_dict["question"][i], "answer": dataset_dict["answer"][i]} 
                        for i in range(total_samples)]
        
        logger.info(f"Processing {total_samples} samples in batches of {BATCH_SIZE}")
        
        # Process batches
        batch_num = 1
        for batch in batch_generator(dataset_list, BATCH_SIZE):
            logger.info(f"Processing batch {batch_num} ({len(batch)} samples)")
            batch_dataset = Dataset.from_list(batch)
            
            try:
                batch_results = evaluate(
                    dataset=batch_dataset,
                    metrics=metrics,
                )
                
                # Merge results - Fix for EvaluationResult object
                # if hasattr(batch_results._repr_dict, 'to_dict'):
                #     # Convert EvaluationResult to dictionary if it has to_dict method
                #     batch_results_dict = batch_results._repr_dict.to_dict()
                #     for metric_name, score in batch_results_dict.items():
                #         if metric_name not in all_results:
                #             all_results[metric_name] = []
                        
                #         if hasattr(score, '__iter__') and not isinstance(score, str):
                #             all_results[metric_name].extend(score)
                #         else:
                #             all_results[metric_name].append(score)
                # else:
                    # For backward compatibility with older RAGAS versions
                for metric_name, score in batch_results._repr_dict.items():
                    if metric_name not in all_results:
                        all_results[metric_name] = []
                    
                    if hasattr(score, '__iter__') and not isinstance(score, str):
                        all_results[metric_name].extend(score)
                    else:
                        all_results[metric_name].append(score)
                        
                # Log progress
                logger.info(f"Completed batch {batch_num}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
            
            # Wait between batches to avoid rate limits
            if batch_num * BATCH_SIZE < total_samples:
                logger.info(f"Waiting {BATCH_DELAY} seconds before next batch...")
                time.sleep(BATCH_DELAY)
            
            batch_num += 1
        
        # Compute average scores
        final_results = {}
        for metric_name, scores in all_results.items():
            if all(isinstance(score, (int, float)) for score in scores):
                final_results[metric_name] = sum(scores) / len(scores)
            else:
                # For non-numeric scores, keep as is
                final_results[metric_name] = scores
        
        # Log final results
        logger.info(f"Evaluation results for {model_name}:")
        for metric_name, score in final_results.items():
            if hasattr(score, '__iter__') and not isinstance(score, str):
                try:
                    avg_score = sum(score) / len(score)
                    logger.info(f"{metric_name}: {avg_score:.4f}")
                except:
                    logger.info(f"{metric_name}: (complex data type)")
            else:
                logger.info(f"{metric_name}: {score:.4f}")
        
        return final_results
    
    except Exception as e:
        logger.error(f"Error during batch evaluation of {model_name}: {e}")
        return {"error": str(e)}

def save_comparison_results(finetuned_results, non_finetuned_results, output_file="model_comparison_results.json"):
    """Save the comparison results to a JSON file"""
    try:
        # Prepare results for serialization
        def prepare_for_json(results):
            if isinstance(results, dict):
                return {k: v.to_dict() if hasattr(v, 'to_dict') else (
                       float(v) if isinstance(v, (int, float)) else v) 
                       for k, v in results.items()}
            return results
        
        finetuned_serializable = prepare_for_json(finetuned_results)
        non_finetuned_serializable = prepare_for_json(non_finetuned_results)
        
        comparison = {
            "finetuned_model": {
                "name": "lora-gpt",
                "results": finetuned_serializable
            },
            "non_finetuned_model": {
                "name": "unsloth/Llama-3.1-8B-Instruct",
                "results": non_finetuned_serializable
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Comparison results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving comparison results: {e}")

def main():
    # Set file paths
    test_data_path = "src/test_data.json"
    finetuned_responses_path = "src/responses_finetuned.json"
    non_finetuned_responses_path = "src/responses_non_finetuned.json"
    
    # Load data
    logger.info("Loading test data and model responses...")
    test_data = load_test_data(test_data_path)
    
    try:
        finetuned_responses = load_model_responses(finetuned_responses_path)
        non_finetuned_responses = load_model_responses(non_finetuned_responses_path)
    except Exception as e:
        logger.error(f"Error loading responses: {e}")
        logger.info("Using fallback method to extract responses...")
        
        # Fallback method: if the response files are incomplete, 
        # use the example output from test_data.json itself
        if 'output' in test_data:
            logger.info("Using example outputs from test_data.json...")
            finetuned_responses = test_data['output']
            non_finetuned_responses = test_data['output']
        else:
            logger.error("No valid responses found for evaluation. Exiting.")
            return
    
    instruction = test_data.get('instruction')[0]
    
    # Get questions from test data and append the instruction before each question
    if 'input' in test_data:
        questions = test_data.get('input', [])
        if questions:
            questions = [f"Instruction: {instruction}\n\nQuestion: {q}" for q in questions]
        else:
            logger.error("No valid input found in test data. Exiting.")
            return
    elif 'instruction' in test_data:
        questions = test_data.get('instruction', [])
    else:
        # Fallback to instruction as questions if input is not available
        questions = test_data.get('instruction', [])[:1]

    min_count = min(len(questions), len(finetuned_responses), len(non_finetuned_responses))
    if min_count == 0:
        logger.error("No valid data for evaluation. Exiting.")
        return
    
    questions = questions[:min_count]
    finetuned_responses = finetuned_responses[:min_count]
    non_finetuned_responses = non_finetuned_responses[:min_count]
    
    logger.info(f"Creating evaluation datasets with {min_count} samples...")
    
    # Define metrics
    metrics = define_metrics()
    
    # Create datasets for evaluation
    finetuned_dataset = create_dataset_for_evaluation(questions, finetuned_responses)
    non_finetuned_dataset = create_dataset_for_evaluation(questions, non_finetuned_responses)
    
    # Run evaluations using batch processing
    logger.info("Evaluating finetuned model (in batches)...")
    finetuned_results = run_evaluation_in_batches(finetuned_dataset, metrics, "Finetuned Model (lora-gpt)")
    
    logger.info("Evaluating non-finetuned model (in batches)...")
    non_finetuned_results = run_evaluation_in_batches(non_finetuned_dataset, metrics, "Non-finetuned Model (unsloth/Llama-3.1-8B-Instruct)")
    
    # Save comparison results
    save_comparison_results(finetuned_results, non_finetuned_results)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()