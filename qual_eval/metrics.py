import os
import requests
import logging
from datasets import Dataset
from ragas.metrics import AspectCritic, RubricsScore
from ragas import evaluate

from qual_eval.constanst import EVAL_QUESTIONS, EMPATHY_CRITERIA, EMPATHY_RUBRIC

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OPEN_API_KEY = os.getenv("OPEN_API_KEY")

# URL of your Modal-hosted LLM endpoint
MODAL_ENDPOINT = os.getenv("MODAL_ENDPOINT")

def call_modal_model(question, endpoint=MODAL_ENDPOINT):
    """
    Calls the Modal-hosted LLM model with the given question and returns its generated response.
    If the call fails for any reason, logs an error and returns an empty string.
    """
    try:
        payload = {"prompt": question}
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        answer = data.get("answer", "")
        return answer
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling model for question '{question}': {e}")
        return ""

def create_evaluation_dataset(questions):
    """
    Generates a list of dictionaries with each dictionary containing:
    'question': the input prompt,
    'answer': the model's generated response.
    """
    eval_data = []
    for question in questions:
        answer = call_modal_model(question)
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
        criteria=EMPATHY_CRITERIA,
        input_key="question",
        prediction_key="answer",
)

    rubric_empathy = RubricsScore(
        name="empathy_score",
        rubric=EMPATHY_RUBRIC,
        input_key="question",
        prediction_key="answer",
    )
    return [empathy_aspect, rubric_empathy]

def run_evaluation(dataset, metrics):
    """
    Runs the RAGAS evaluation given the dataset and list of metrics.
    Returns a dictionary of the evaluation scores.
    """
    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
    )
    return scores

def main():
    logger.info("Collecting responses from the Modal-hosted LLM...")
    eval_data = create_evaluation_dataset(EVAL_QUESTIONS)
    
    # Convert list of dictionaries to a HuggingFace Dataset object
    eval_dataset = Dataset.from_list(eval_data)

    logger.info("Defining empathy evaluation metrics (AspectCritic and RubricScore)...")
    empathy_metrics = define_empathy_metrics()

    logger.info("Running evaluation...")
    empathy_scores = run_evaluation(eval_dataset, empathy_metrics)

    logger.info("Evaluation Results:")
    for metric_name, score in empathy_scores.items():
        logger.info(f"{metric_name}: {score}")

if __name__ == "__main__":
    main()