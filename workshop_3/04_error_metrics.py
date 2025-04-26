import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import seaborn as sns

# Path to the evaluation report
REPORT_PATH = Path(__file__).parent / "data" / "evaluation_report.json"
ERROR_ANALYSIS_PATH = Path(__file__).parent / "data" / "error_analysis.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

def load_evaluation_report():
    """Load the evaluation report from the JSON file."""
    with open(REPORT_PATH, 'r') as f:
        report = json.load(f)
    return report

def load_error_analysis():
    """Load the error analysis report if it exists."""
    try:
        with open(ERROR_ANALYSIS_PATH, 'r') as f:
            analysis = json.load(f)
        return analysis
    except FileNotFoundError:
        print(f"Error analysis file not found at {ERROR_ANALYSIS_PATH}")
        return None
    except json.JSONDecodeError:
        print(f"Error analysis file at {ERROR_ANALYSIS_PATH} is not valid JSON")
        return None

def display_overall_metrics(report):
    """Display overall metrics in a table format."""
    overall_metrics = report["overall_metrics"]
    
    # Format the metrics for display
    metrics_data = [
        ["Metric", "Value"],
        ["Accuracy", f"{overall_metrics['accuracy']:.4f}"],
        ["Precision", f"{overall_metrics['precision']:.4f}"],
        ["Recall", f"{overall_metrics['recall']:.4f}"],
        ["F1 Score", f"{overall_metrics['f1']:.4f}"]
    ]
    
    # Display the table
    print("\n===== Overall Evaluation Metrics =====")
    print(tabulate(metrics_data, headers="firstrow", tablefmt="grid"))
    
    # Save to CSV
    metrics_df = pd.DataFrame(overall_metrics, index=[0])
    metrics_df.to_csv(OUTPUT_DIR / "overall_metrics.csv", index=False)
    print(f"Overall metrics saved to {OUTPUT_DIR / 'overall_metrics.csv'}")
    
    # Generate a nicer HTML version
    html_content = f"""
    <html>
    <head>
        <title>RAG Evaluation - Overall Metrics</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metrics-table {{ border-collapse: collapse; width: 50%; margin: 20px 0; }}
            .metrics-table th, .metrics-table td {{ 
                border: 1px solid #ddd; padding: 12px; text-align: center; 
            }}
            .metrics-table th {{ background-color: #f2f2f2; }}
            h1 {{ color: #333; }}
            .header {{ background-color: #4CAF50; color: white; padding: 10px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>RAG Evaluation Results</h1>
        </div>
        <h2>Overall Metrics</h2>
        <table class="metrics-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Accuracy</td>
                <td>{overall_metrics['accuracy']:.4f}</td>
            </tr>
            <tr>
                <td>Precision</td>
                <td>{overall_metrics['precision']:.4f}</td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>{overall_metrics['recall']:.4f}</td>
            </tr>
            <tr>
                <td>F1 Score</td>
                <td>{overall_metrics['f1']:.4f}</td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    with open(OUTPUT_DIR / "overall_metrics.html", 'w') as f:
        f.write(html_content)
    print(f"Overall metrics HTML report saved to {OUTPUT_DIR / 'overall_metrics.html'}")
    
    return pd.DataFrame(overall_metrics, index=[0])

def display_question_metrics(report):
    """Display per-question metrics in a table format."""
    question_metrics = report["question_metrics"]
    
    # Convert to DataFrame for easier handling
    questions = []
    accuracies = []
    counts = []
    
    for question, metrics in question_metrics.items():
        questions.append(question)
        accuracies.append(metrics["accuracy"])
        counts.append(metrics["count"])
    
    # Create DataFrame
    df = pd.DataFrame({
        "Question": questions,
        "Accuracy": accuracies,
        "Count": counts
    })
    
    # Sort by accuracy (descending)
    df = df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    # Format for display
    display_df = df.copy()
    display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x:.4f}")
    
    # Display the table
    print("\n===== Metrics by Question Type =====")
    print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=False))
    
    # Save to CSV
    df.to_csv(OUTPUT_DIR / "question_metrics.csv", index=False)
    print(f"Question metrics saved to {OUTPUT_DIR / 'question_metrics.csv'}")
    
    return df

def display_detailed_results(report):
    """Display detailed results in a table format."""
    detailed_results = report["detailed_results"]
    
    # Truncate long texts for better display
    table_data = []
    headers = ["Question", "Resume", "Gold Answer", "Predicted", "Correct"]
    
    for result in detailed_results:
        question = result["question"]
        resume = result["resume"].split('/')[-1]  # Just the filename
        gold_answer = result["gold_answer"][:50] + "..." if len(result["gold_answer"]) > 50 else result["gold_answer"]
        pred_answer = result["predicted_answer"][:50] + "..." if len(result["predicted_answer"]) > 50 else result["predicted_answer"]
        is_correct = "✓" if result["is_correct"] else "✗"
        
        table_data.append([question, resume, gold_answer, pred_answer, is_correct])
    
    # Display the table
    print("\n===== Detailed Evaluation Results =====")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def plot_question_metrics(df):
    """Create a horizontal bar chart of question accuracies."""
    plt.figure(figsize=(10, 8))
    
    # Sort by accuracy
    df = df.sort_values(by="Accuracy")
    
    # Truncate long question text for better display
    df['Question_Short'] = df['Question'].apply(lambda x: x[:40] + '...' if len(x) > 40 else x)
    
    # Create horizontal bar chart
    bars = plt.barh(df['Question_Short'], df['Accuracy'], color='skyblue')
    
    # Add value labels to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(max(width + 0.02, 0.02),  # Shift text position slightly
                 bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}',
                 va='center')
    
    plt.xlabel('Accuracy')
    plt.ylabel('Question')
    plt.title('RAG Accuracy by Question Type')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlim(0, 1.1)  # Set x-axis limit to accommodate the text
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "question_accuracy.png")
    print(f"\nPlot saved to {OUTPUT_DIR / 'question_accuracy.png'}")

def plot_correct_vs_incorrect(report):
    """Create a pie chart showing correct vs incorrect answers."""
    detailed_results = report["detailed_results"]
    
    # Count correct and incorrect answers
    correct = sum(1 for result in detailed_results if result.get("is_correct", False))
    incorrect = len(detailed_results) - correct
    
    # Create pie chart
    plt.figure(figsize=(8, 8))
    plt.pie([correct, incorrect], 
            labels=['Correct', 'Incorrect'],
            colors=['#4CAF50', '#F44336'],
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.05, 0))
    
    plt.title('RAG Answer Accuracy')
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correct_vs_incorrect.png")
    print(f"Plot saved to {OUTPUT_DIR / 'correct_vs_incorrect.png'}")

def display_error_analysis(error_analysis):
    """Display error analysis insights."""
    if not error_analysis or "error_summary" not in error_analysis:
        print("\nNo error analysis data available.")
        return
    
    error_summary = error_analysis["error_summary"]
    
    # Display key insights
    print("\n===== Error Analysis Key Insights =====")
    for insight in error_summary.get("key_insights", []):
        print(f"- {insight}")
    
    # Display error types
    error_types = error_summary.get("by_type", {})
    if error_types:
        print("\n===== Error Types =====")
        error_type_data = [[error_type, count] for error_type, count in error_types.items()]
        error_type_data.sort(key=lambda x: x[1], reverse=True)
        error_type_data.insert(0, ["Error Type", "Count"])
        print(tabulate(error_type_data, headers="firstrow", tablefmt="grid"))
    
    # Display error causes
    error_causes = error_summary.get("by_cause", {})
    if error_causes:
        print("\n===== Error Causes =====")
        error_cause_data = [[cause, count] for cause, count in error_causes.items()]
        error_cause_data.sort(key=lambda x: x[1], reverse=True)
        error_cause_data.insert(0, ["Likely Cause", "Count"])
        print(tabulate(error_cause_data, headers="firstrow", tablefmt="grid"))
    
    # Save to CSV
    if error_types and error_causes:
        # Create DataFrames
        types_df = pd.DataFrame(list(error_types.items()), columns=["Error Type", "Count"])
        causes_df = pd.DataFrame(list(error_causes.items()), columns=["Likely Cause", "Count"])
        
        # Save to CSV
        types_df.to_csv(OUTPUT_DIR / "error_types.csv", index=False)
        causes_df.to_csv(OUTPUT_DIR / "error_causes.csv", index=False)
        print(f"Error analysis data saved to {OUTPUT_DIR}")
    
    # Generate HTML error analysis report
    generate_error_analysis_html(error_analysis)

def generate_error_analysis_html(error_analysis):
    """Generate a detailed HTML report for error analysis."""
    if not error_analysis:
        return
    
    error_summary = error_analysis["error_summary"]
    detailed_analyses = error_analysis.get("detailed_error_analyses", [])
    
    # Start building HTML content
    html_content = """
    <html>
    <head>
        <title>RAG Error Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #F44336; color: white; padding: 10px; }
            h1 { color: white; }
            h2 { color: #333; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .insights { background-color: #f9f9f9; padding: 15px; border-radius: 5px; }
            .error-card { 
                border: 1px solid #ddd; 
                margin: 15px 0; 
                padding: 15px; 
                border-radius: 5px;
                background-color: #fff;
            }
            .critical { border-left: 5px solid darkred; }
            .major { border-left: 5px solid orangered; }
            .minor { border-left: 5px solid gold; }
            .error-header {
                display: flex;
                justify-content: space-between;
            }
            .error-type {
                font-weight: bold;
                color: #333;
            }
            .severity {
                font-weight: bold;
            }
            .critical-text { color: darkred; }
            .major-text { color: orangered; }
            .minor-text { color: goldenrod; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>RAG Error Analysis Report</h1>
        </div>
    """
    
    # Add key insights
    html_content += """
        <h2>Key Insights</h2>
        <div class="insights">
        <ul>
    """
    
    for insight in error_summary.get("key_insights", []):
        html_content += f"<li>{insight}</li>\n"
    
    html_content += """
        </ul>
        </div>
    """
    
    # Add error type statistics
    error_types = error_summary.get("by_type", {})
    if error_types:
        html_content += """
        <h2>Error Types</h2>
        <table>
            <tr>
                <th>Error Type</th>
                <th>Count</th>
            </tr>
        """
        
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            html_content += f"""
            <tr>
                <td>{error_type}</td>
                <td>{count}</td>
            </tr>
            """
        
        html_content += "</table>"
    
    # Add error causes statistics
    error_causes = error_summary.get("by_cause", {})
    if error_causes:
        html_content += """
        <h2>Error Causes</h2>
        <table>
            <tr>
                <th>Likely Cause</th>
                <th>Count</th>
            </tr>
        """
        
        for cause, count in sorted(error_causes.items(), key=lambda x: x[1], reverse=True):
            html_content += f"""
            <tr>
                <td>{cause}</td>
                <td>{count}</td>
            </tr>
            """
        
        html_content += "</table>"
    
    # Add detailed error analyses
    if detailed_analyses:
        html_content += """
        <h2>Detailed Error Analyses</h2>
        """
        
        for analysis in detailed_analyses:
            severity_class = analysis["severity"].lower()
            severity_text_class = f"{severity_class}-text"
            
            html_content += f"""
            <div class="error-card {severity_class}">
                <div class="error-header">
                    <span class="error-type">{analysis["error_type"]}</span>
                    <span class="severity {severity_text_class}">Severity: {analysis["severity"]}</span>
                </div>
                
                <h3>Question: {analysis["question"]}</h3>
                <p><strong>Resume:</strong> {analysis["resume"]}</p>
                
                <p><strong>Gold Answer:</strong> {analysis["gold_answer"]}</p>
                <p><strong>Predicted Answer:</strong> {analysis["predicted_answer"]}</p>
                
                <p><strong>Details:</strong> {analysis["details"]}</p>
                <p><strong>Likely Cause:</strong> {analysis["likely_cause"]}</p>
                <p><strong>Fix Recommendation:</strong> {analysis["fix_recommendation"]}</p>
            </div>
            """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write to file
    with open(OUTPUT_DIR / "error_analysis.html", 'w') as f:
        f.write(html_content)
    print(f"Detailed error analysis HTML report saved to {OUTPUT_DIR / 'error_analysis.html'}")

def main():
    """Main function to run the display metrics script."""
    try:
        # Load the evaluation report
        report = load_evaluation_report()
        
        # Load error analysis if available
        error_analysis = load_error_analysis()
        
        # Display overall metrics
        overall_df = display_overall_metrics(report)
        
        # Display per-question metrics
        question_df = display_question_metrics(report)
        
        # Display detailed results
        display_detailed_results(report)
        
        # Plot question metrics
        plot_question_metrics(question_df)
        
        # Plot correct vs incorrect
        plot_correct_vs_incorrect(report)
        
        # Display error analysis if available
        if error_analysis:
            display_error_analysis(error_analysis)
        
        print("\nMetrics display completed successfully.")
        
    except FileNotFoundError:
        print(f"Error: Could not find the evaluation report at {REPORT_PATH}")
    except json.JSONDecodeError:
        print(f"Error: The evaluation report at {REPORT_PATH} is not valid JSON")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()