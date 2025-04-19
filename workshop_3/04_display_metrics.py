import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tabulate import tabulate

# Path to the evaluation report
REPORT_PATH = Path(__file__).parent / "data" / "evaluation_report.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

def load_evaluation_report():
    """Load the evaluation report from the JSON file."""
    with open(REPORT_PATH, 'r') as f:
        report = json.load(f)
    return report

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

def main():
    """Main function to run the display metrics script."""
    try:
        # Load the evaluation report
        report = load_evaluation_report()
        
        # Display overall metrics
        overall_df = display_overall_metrics(report)
        
        # Display per-question metrics
        question_df = display_question_metrics(report)
        
        # Plot question metrics
        plot_question_metrics(question_df)
        
        # Plot correct vs incorrect
        plot_correct_vs_incorrect(report)
        
        print("\nMetrics display completed successfully.")
        
    except FileNotFoundError:
        print(f"Error: Could not find the evaluation report at {REPORT_PATH}")
    except json.JSONDecodeError:
        print(f"Error: The evaluation report at {REPORT_PATH} is not valid JSON")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()