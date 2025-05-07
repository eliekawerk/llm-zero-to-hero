import json
import os
import pandas as pd
from tqdm import tqdm
import instructor
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv(verbose=True, dotenv_path=".env")

# Ensure you have set GOOGLE_API_KEY in your environment
if not os.getenv("GOOGLE_API_KEY"):
  raise ValueError("Please set GOOGLE_API_KEY environment variable")

google_api_key = os.getenv("GOOGLE_API_KEY")
# print(f"Google API Key: {google_api_key}")
genai.configure(api_key=google_api_key)

# Initialize the Gemini client with Instructor
client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-latest",  
    )
)

# Define paths
GOLD_SET_PATH = "data/gold_set.jsonl"
PREDICTIONS_PATH = "data/your_predictions.jsonl"
OUTPUT_REPORT_PATH = "data/evaluation_report.json"
ERROR_ANALYSIS_PATH = "data/error_analysis.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Pydantic model for evaluation
class AnswerEvaluation(BaseModel):
  """Evaluation of an answer against a gold standard."""
  is_correct: bool = Field(..., description="Whether the predicted answer is correct based on the gold standard")
  percentage_correct: float = Field(..., ge=0, le=1, description="Percentage of correctness in the prediction")
  reasoning: str = Field(..., description="Reasoning behind the evaluation decision")
  missing_info: Optional[List[str]] = Field(None, description="Important information missing from the prediction")
  incorrect_info: Optional[List[str]] = Field(None, description="Incorrect information in the prediction")

# Pydantic model for error analysis
class ErrorAnalysis(BaseModel):
  """Detailed error analysis of an incorrect prediction"""
  error_type: Literal[
    "missing_information", 
    "incorrect_information", 
    "hallucination",
    "ambiguity",
    "formatting_issue",
    "other"
  ] = Field(..., description="The primary type of error")
  
  severity: Literal["critical", "major", "minor"] = Field(
    ..., description="How severely this error impacts the usefulness of the response"
  )
  
  details: str = Field(..., description="Detailed explanation of the error")
  
  likely_cause: Literal[
    "retrieval_failure", 
    "chunking_issue", 
    "context_window_limitation",
    "prompt_misalignment",
    "embedding_similarity_failure",
    "llm_reasoning_error",
    "other"
  ] = Field(..., description="The most likely cause of this error")
  
  fix_recommendation: str = Field(..., description="Recommendation on how to fix this error")

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
    
    Evaluate if the predicted answer is semantically equivalent to the gold standard answer,
    and base your evaluation on the following criteria:
    1. The predicted answer must contain the same key information as the gold standard
    2. Minor differences in wording are acceptable as long as the meaning is preserved
    3. The percentage correct conveys the estimate of correctness for the prediction. It is a float
    between 0.0 and 1.0, and the following explains the scale:
    - 0.76 to 1.0 means the prediction is 76% to 100% correct
    - 0.51 to 0.75 means the prediction is 51% to 75% correct
    - 0.26 to 0.5 means the prediction is 26% to 50% correct
    - 0.01 to 0.25 means the prediction is 1% to 25% correct
    - 0.0 means the prediction is completely incorrect
    4. **Set is_correct to True if the percentage correct is 0.5 to 1.0**, else set it to False. This is the threshold for correctness.
    5. The predicted answer should not contain significant incorrect information
    6. If the gold standard states "Not mentioned in the resume", the prediction should indicate the information is not available
    7. The predicted answer should not contain hallucinations or irrelevant information
    8. Lastly, remeber that value in is_correct will depend on the value of the threshold for correctness.
    
    Provide your evaluation with clear reasoning.
    """
    
    evaluation = client.chat.completions.create(
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
      percentage_correct=0.0,
      reasoning=f"Evaluation failed: {str(e)}",
      missing_info=None,
      incorrect_info=None
    )

def analyze_error(gold_answer, predicted_answer, question):
  """Perform detailed error analysis on incorrect predictions."""
  try:
    prompt = f"""
    Question: "{question}"
    Gold standard answer: "{gold_answer}"
    Incorrect predicted answer: "{predicted_answer}"
    
    Analyze why this prediction is incorrect and categorize the error.
    Identify:
    1. The primary type of error (missing information, incorrect information, hallucination, etc.)
    2. How severe this error is (critical, major, minor)
    3. Detailed explanation of what went wrong
    4. The likely cause of this error in the RAG pipeline (retrieval failure, chunking issue, etc.)
    5. Recommendation on how to fix this error
    """
    
    analysis = client.chat.completions.create(
      response_model=ErrorAnalysis,
      messages=[
        {"role": "system", "content": "You are an expert evaluator of RAG systems who specializes in error analysis."},
        {"role": "user", "content": prompt}
      ]
    )
    return analysis
  except Exception as e:
    print(f"Error in analysis: {e}")
    return ErrorAnalysis(
      error_type="other",
      severity="major",
      details=f"Error analysis failed: {str(e)}",
      likely_cause="other",
      fix_recommendation="Retry the error analysis"
    )

def analyze_errors(results, merged_df):
  """Analyze all errors and categorize them."""
  error_analyses = []
  
  # Find all incorrect predictions
  incorrect_results = [(i, r) for i, r in enumerate(results) if not r['is_correct']]
  
  if incorrect_results:
    print(f"\nAnalyzing {len(incorrect_results)} incorrect predictions...")
    
    for i, result in tqdm(incorrect_results, desc="Analyzing errors"):
      row_index = merged_df.index[i]
      question = merged_df.loc[row_index, 'question']
      gold_answer = result['gold_answer']
      pred_answer = result['predicted_answer']
      
      # Perform detailed error analysis
      error_analysis = analyze_error(gold_answer, pred_answer, question)
      
      # Add to results
      analysis_dict = error_analysis.model_dump()
      analysis_dict['resume'] = result['resume']
      analysis_dict['question'] = result['question']
      analysis_dict['gold_answer'] = gold_answer
      analysis_dict['predicted_answer'] = pred_answer
      analysis_dict['is_correct'] = result['is_correct']
      analysis_dict['percentage_correct'] = result['percentage_correct']
      analysis_dict['reasoning'] = result['reasoning']
      error_analyses.append(analysis_dict)
  
  return error_analyses

def generate_error_summary(error_analyses, results):
  """Generate summary statistics from error analyses."""
  if not error_analyses:
    return {"message": "No errors to analyze"}
  
  # Count errors by type
  error_types = Counter([e['error_type'] for e in error_analyses])
  
  # Count errors by severity
  error_severity = Counter([e['severity'] for e in error_analyses])
  
  # Count likely causes
  error_causes = Counter([e['likely_cause'] for e in error_analyses])
  
  # Count errors by question
  error_questions = Counter([e['question'] for e in error_analyses])
  
  # Generate key insights
  key_insights = []
  
  # Most common error type
  if error_types:
    most_common_type = max(error_types.items(), key=lambda x: x[1])
    key_insights.append(f"Most common error type: {most_common_type[0]} ({most_common_type[1]} occurrences)")
  
  # Most common cause
  if error_causes:
    most_common_cause = max(error_causes.items(), key=lambda x: x[1])
    key_insights.append(f"Most common cause: {most_common_cause[0]} ({most_common_cause[1]} occurrences)")
  
  # Most problematic question
  if error_questions:
    most_problematic = max(error_questions.items(), key=lambda x: x[1])
    key_insights.append(f"Most problematic question: '{most_problematic[0]}' ({most_problematic[1]} failures)")
  
  # Critical errors
  critical_count = error_severity.get('critical', 0)
  if critical_count > 0:
    critical_pct = critical_count / len(error_analyses) * 100
    key_insights.append(f"Critical errors: {critical_count} ({critical_pct:.1f}% of all errors)")
  
  return {
    "total_samples": len(results),
    "error_count": len(error_analyses),
    "by_type": dict(error_types),
    "by_severity": dict(error_severity),
    "by_cause": dict(error_causes),
    "by_question": dict(error_questions),
    "key_insights": key_insights
  }

def plot_error_analysis(error_summary, output_dir):
  """Create visualizations of error analysis."""
  if "message" in error_summary:
    return  # No errors to plot
  
  # Set the style
  sns.set_style("whitegrid")
  
  # Plot error types
  plt.figure(figsize=(10, 6))
  sns.barplot(x=list(error_summary["by_type"].keys()), 
              y=list(error_summary["by_type"].values()),
              palette="viridis")
  plt.title("Errors by Type")
  plt.xticks(rotation=45, ha='right')
  plt.tight_layout()
  plt.savefig(output_dir / "error_types.png")
  
  # Plot error causes
  plt.figure(figsize=(12, 6))
  sns.barplot(x=list(error_summary["by_cause"].keys()), 
              y=list(error_summary["by_cause"].values()),
              palette="mako")
  plt.title("Errors by Likely Cause")
  plt.xticks(rotation=45, ha='right')
  plt.tight_layout()
  plt.savefig(output_dir / "error_causes.png")
  
  # Plot error severity
  plt.figure(figsize=(8, 6))
  sizes = list(error_summary["by_severity"].values())
  labels = list(error_summary["by_severity"].keys())
  colors = {"critical": "darkred", "major": "orangered", "minor": "gold"}
  plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
          colors=[colors[sev] for sev in error_summary["by_severity"].keys()])
  plt.axis('equal')
  plt.title("Error Severity Distribution")
  plt.tight_layout()
  plt.savefig(output_dir / "error_severity.png")
  
def adjust_by_threshold(result, threshold=0.5):
    result.is_correct = result.percentage_correct >= threshold
    return result

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
    
    # Evaluate using Instructor and adjust by threshold
    evaluation = adjust_by_threshold(evaluate_answer(gold_answer, pred_answer), threshold=0.75)
    # print(f"Evaluation: {evaluation}")
    
    # Store results
    result = {
      'resume': row['resume'],
      'question': row['question'],
      'gold_answer': gold_answer,
      'predicted_answer': pred_answer,
      'is_correct': evaluation.is_correct,
      'percentage_correct': evaluation.percentage_correct,
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
    'total_samples': len(results),
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
  
  # Analyze errors in detail
  error_analyses = analyze_errors(results, merged_df)
  
  # Generate error summary
  error_summary = generate_error_summary(error_analyses, results)
  
  # Plot error analysis visualizations
  plot_error_analysis(error_summary, OUTPUT_DIR)
  
  # Compile full report
  report = {
    'overall_metrics': metrics,
    'question_metrics': question_metrics,
    'detailed_results': results,
    'error_summary': error_summary
  }
  
  # Save reports
  with open(OUTPUT_REPORT_PATH, 'w') as f:
    json.dump(report, f, indent=2)
  
  if error_analyses:
    with open(ERROR_ANALYSIS_PATH, 'w') as f:
      json.dump({
        "error_summary": error_summary,
        "detailed_error_analyses": error_analyses
      }, f, indent=2)
    print(f"\nError analysis saved to {ERROR_ANALYSIS_PATH}")
    print("\nError Analysis Summary:")
    for insight in error_summary["key_insights"]:
      print(f"- {insight}")
  
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
  print(f"Error visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
  main()