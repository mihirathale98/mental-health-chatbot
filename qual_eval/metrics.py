import os
import logging
import json
import time
from datasets import Dataset
from ragas.metrics import AspectCritic, RubricsScore
from ragas import evaluate
from openai import OpenAI

# Fix typo in import statement
from constanst import EVAL_QUESTIONS, EMPATHY_CRITERIA, EMPATHY_RUBRIC

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini')

def call_openai_model(question):
    """
    Calls the OpenAI API with the given question and returns its generated response.
    If the call fails for any reason, logs an error and returns an empty string.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that responds with empathy."},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        # Extract the text from the response
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        logger.error(f"Error calling OpenAI model for question '{question[:50]}...': {e}")
        return ""

def create_evaluation_dataset(questions):
    """
    Generates a list of dictionaries with each dictionary containing:
    'question': the input prompt,
    'answer': the model's generated response.
    """
    eval_data = []
    for i, question in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
        answer = call_openai_model(question)
        eval_data.append({
            "question": question,
            "answer": answer
        })
    return eval_data

def define_empathy_metrics():
    """
    Creates the AspectCritic and RubricScore metrics for evaluating empathy.
    """
    empathy_aspect = AspectCritic(
        name="empathy_check",
        definition=EMPATHY_CRITERIA,
        strictness=0.8
    )

    # Fix: Convert EMPATHY_RUBRIC tuple to the format RubricsScore expects
    if isinstance(EMPATHY_RUBRIC, tuple):
        # Convert tuple to dictionary with numbered keys
        rubrics_dict = {f"rubric_{i}": rubric for i, rubric in enumerate(EMPATHY_RUBRIC)}
        
        rubric_empathy = RubricsScore(
            name="empathy_score",
            rubrics=rubrics_dict,
        )
    else:
        # If it's already a dictionary, use it directly
        rubric_empathy = RubricsScore(
            name="empathy_score",
            rubrics=EMPATHY_RUBRIC,
        )
        
    return [empathy_aspect, rubric_empathy]

def run_evaluation(dataset, metrics):
    """
    Runs the RAGAS evaluation given the dataset and list of metrics.
    Returns the evaluation results dictionary.
    """
    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
        )
        return results
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return {"error": str(e)}

def save_results(eval_data, evaluation_results, output_file="empathy_evaluation_results.json"):
    """
    Saves both the raw responses and evaluation scores to a JSON file.
    This allows for later analysis of individual responses.
    """
    try:
        # Create a serializable dictionary from the evaluation results
        scores_dict = {}
        
        # If evaluation_results is already a dictionary or dictionary-like
        if hasattr(evaluation_results, 'items'):
            # Convert each item to a serializable format
            for k, v in evaluation_results.items():
                if hasattr(v, 'to_dict'):  # For pandas objects
                    scores_dict[k] = v.to_dict()
                else:
                    scores_dict[k] = v
        else:
            # Fallback for non-dictionary objects
            scores_dict = {"result": str(evaluation_results)}
            
        results = {
            "raw_data": eval_data,
            "scores": scores_dict,
            "model_used": MODEL_NAME,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")
        fallback_file = f"empathy_evaluation_results_fallback_{int(time.time())}.json"
        with open(fallback_file, 'w') as f:
            json.dump({"error": str(e), "raw_data": eval_data}, f, indent=2)
        logger.info(f"Error details saved to fallback file: {fallback_file}")

def main():
    logger.info("Starting empathy evaluation with OpenAI API...")
    
    # Validate environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        return
    
    logger.info(f"Using model: {MODEL_NAME}")
    
    # Validate that we have questions to evaluate
    if not EVAL_QUESTIONS:
        logger.error("No evaluation questions found in constants!")
        return
    
    logger.info(f"Found {len(EVAL_QUESTIONS)} questions for evaluation")
    
    logger.info("Collecting responses from OpenAI API...")
    eval_data = create_evaluation_dataset(EVAL_QUESTIONS)
    
    # Convert list of dictionaries to a HuggingFace Dataset object
    try:
        eval_dataset = Dataset.from_list(eval_data)
    except Exception as e:
        logger.error(f"Error creating Dataset object: {e}")
        return
    
    logger.info("Defining empathy evaluation metrics (AspectCritic and RubricScore)...")
    empathy_metrics = define_empathy_metrics()

    logger.info("Running evaluation...")
    try:
        evaluation_results = run_evaluation(eval_dataset, empathy_metrics)
        
        logger.info("Evaluation Results:")
        # Properly handle the results based on the output we saw
        # The results appear to be a dictionary-like object with metric names as keys
        if hasattr(evaluation_results, 'items'):
            for metric_name, score in evaluation_results.items():
                logger.info(f"{metric_name}: {score:.4f}")
        else:
            logger.info(f"Raw evaluation result: {evaluation_results}")
            
        # Save results for later analysis
        save_results(eval_data, evaluation_results)
        
    except Exception as e:
        logger.error(f"Error in evaluation process: {e}")
        
    logger.info("Evaluation process completed")

if __name__ == "__main__":
    main()