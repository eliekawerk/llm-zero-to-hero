import json
import os
import pandas as pd
from tqdm import tqdm
import instructor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure you have set OPENAI_API_KEY in your environment
if not os.getenv("OPENAI_API_KEY"):
  raise ValueError("Please set OPENAI_API_KEY environment variable")

# Initialize the OpenAI client with Instructor
client = instructor.patch(OpenAI())

# Define paths
GOLD_SET_PATH = "data/gold_set.jsonl"
PREDICTIONS_PATH = "data/your_predictions.jsonl"
OUTPUT_REPORT_PATH = "data/evaluation_report.json"

# Pydantic model for evaluation
class AnswerEvaluation(BaseModel):
  """Evaluation of an answer against a gold standard."""
  is_correct: bool = Field(..., description="Whether the predicted answer is correct based on the gold standard")
  reasoning: str = Field(..., description="Reasoning behind the evaluation decision")
  missing_info: Optional[List[str]] = Field(None, description="Important information missing from the prediction")
  incorrect_info: Optional[List[str]] = Field(None, description="Incorrect information in the prediction")

def load_jsonl(file_path):
  """Load data from a JSONL file."""
  data = []
  with open(file_path, 'r') as f:
    for line in f:
      data.append(json.loads(line))
  return data

def evaluate_answer(gold_answer, predicted_answer):
  """Evaluate a predicted answer against a gold standard using Instructor."""
  try:
    prompt = f"""
    Gold standard answer: "{gold_answer}"
    Predicted answer: "{predicted_answer}"
    
    Evaluate if the predicted answer is semantically equivalent to the gold standard answer.
    Consider:
    1. The predicted answer must contain the same key information as the gold standard
    2. Minor differences in wording are acceptable as long as the meaning is preserved
    3. The predicted answer should not contain significant incorrect information
    4. If the gold standard states "Not mentioned in the resume", the prediction should indicate the information is not available
    
    Provide your evaluation with clear reasoning.
    """
    
    evaluation = client.chat.completions.create(
      model="gpt-4o",
      response_model=AnswerEvaluation,
      messages=[
        {"role": "system", "content": "You are an expert evaluator of question answering systems."},
        {"role": "user", "content": prompt}
      ]
    )
    return evaluation
  except Exception as e:
    print(f"Error in evaluation: {e}")
    return AnswerEvaluation(
      is_correct=False,
      reasoning=f"Evaluation failed: {str(e)}",
      missing_info=None,
      incorrect_info=None
    )

def main():
  # Load gold standard and predictions
  gold_data = load_jsonl(GOLD_SET_PATH)
  pred_data = load_jsonl(PREDICTIONS_PATH)
  
  print(f"Loaded {len(gold_data)} gold standard items and {len(pred_data)} predictions")
  
  # Convert to DataFrame for easier manipulation
  gold_df = pd.DataFrame(gold_data)
  pred_df = pd.DataFrame(pred_data)
  
  # Merge datasets on resume and question
  merged_df = pd.merge(gold_df, pred_df, on=['resume', 'question'], how='inner')
  
  print(f"Found {len(merged_df)} matching question-resume pairs")
  
  # Initialize results
  results = []
  y_true = []
  y_pred = []
  
  # Evaluate each prediction
  for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Evaluating predictions"):
    gold_answer = row['expected_answer']
    pred_answer = row['predicted_answer']
    
    # Evaluate using Instructor
    evaluation = evaluate_answer(gold_answer, pred_answer)
    
    # Store results
    result = {
      'resume': row['resume'],
      'question': row['question'],
      'gold_answer': gold_answer,
      'predicted_answer': pred_answer,
      'is_correct': evaluation.is_correct,
      'reasoning': evaluation.reasoning
    }
    
    if evaluation.missing_info:
      result['missing_info'] = evaluation.missing_info
    
    if evaluation.incorrect_info:
      result['incorrect_info'] = evaluation.incorrect_info
    
    results.append(result)
    y_true.append(1)  # Gold standard is always considered correct (1)
    y_pred.append(1 if evaluation.is_correct else 0)
  
  # Calculate metrics
  metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'f1': f1_score(y_true, y_pred)
  }
  
  # Add per-question metrics
  question_metrics = {}
  for question in merged_df['question'].unique():
    question_results = [r['is_correct'] for i, r in enumerate(results) if merged_df.iloc[i]['question'] == question]
    
    question_metrics[question] = {
      'accuracy': sum(question_results) / len(question_results) if question_results else 0,
      'count': len(question_results)
    }
  
  # Compile full report
  report = {
    'overall_metrics': metrics,
    'question_metrics': question_metrics,
    'detailed_results': results
  }
  
  # Save report
  with open(OUTPUT_REPORT_PATH, 'w') as f:
    json.dump(report, f, indent=2)
  
  # Print summary
  print("\nEvaluation Summary:")
  print(f"Total evaluated: {len(results)}")
  print(f"Correct: {sum(y_pred)}")
  print(f"Incorrect: {len(y_pred) - sum(y_pred)}")
  print(f"Accuracy: {metrics['accuracy']:.4f}")
  print(f"Precision: {metrics['precision']:.4f}")
  print(f"Recall: {metrics['recall']:.4f}")
  print(f"F1 Score: {metrics['f1']:.4f}")
  print(f"\nDetailed report saved to {OUTPUT_REPORT_PATH}")

if __name__ == "__main__":
  main()