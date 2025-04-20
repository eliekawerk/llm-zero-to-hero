import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from tabulate import tabulate

# Define paths
PREDICTIONS_PATH = Path(__file__).parent / "data" / "your_predictions.jsonl"
EVALUATION_REPORT_PATH = Path(__file__).parent / "data" / "evaluation_report.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# OpenAI pricing (as of April 2025, update as needed)
PRICING = {
    "gpt-4o": {
        "input": 0.01,  # per 1K tokens
        "output": 0.03  # per 1K tokens
    },
    "embedding": {
        "ada": 0.0001  # per 1K tokens
    }
}

def load_data():
    """Load prediction data and evaluation results"""
    # Load predictions with metrics
    predictions = []
    with open(PREDICTIONS_PATH, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    
    # Load evaluation report
    with open(EVALUATION_REPORT_PATH, 'r') as f:
        evaluation = json.load(f)
    
    # Convert predictions to DataFrame
    df = pd.DataFrame(predictions)
    
    # Add correctness from evaluation results
    detailed_results = evaluation.get("detailed_results", [])
    correctness = {}
    
    for result in detailed_results:
        key = (result["resume"], result["question"])
        correctness[key] = result["is_correct"]
    
    # Map correctness to the DataFrame
    df["is_correct"] = df.apply(
        lambda row: correctness.get((row["resume"], row["question"]), False), 
        axis=1
    )
    
    return df, evaluation

def calculate_cost_metrics(df):
    """Calculate cost-related metrics"""
    # Fill any missing token values with 0
    for col in ["input_tokens", "output_tokens", "total_tokens"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0
    
    # Calculate costs
    df["input_cost"] = df["input_tokens"] * PRICING["gpt-4o"]["input"] / 1000
    df["output_cost"] = df["output_tokens"] * PRICING["gpt-4o"]["output"] / 1000
    df["total_cost"] = df["input_cost"] + df["output_cost"]
    
    # Calculate cost per token
    df["cost_per_token"] = df.apply(
        lambda row: row["total_cost"] / row["total_tokens"] if row["total_tokens"] > 0 else 0,
        axis=1
    )
    
    # Calculate efficiency metrics
    df["tokens_per_correct_answer"] = df.apply(
        lambda row: row["total_tokens"] if row["is_correct"] else 0,
        axis=1
    )
    
    # Calculate accuracy per 1K tokens (for groups)
    df["accuracy_per_1k_tokens"] = df.apply(
        lambda row: 1000 * int(row["is_correct"]) / row["total_tokens"] if row["total_tokens"] > 0 else 0,
        axis=1
    )
    
    return df

def analyze_metrics_by_question(df):
    """Group and analyze metrics by question type"""
    # Group by question
    question_metrics = df.groupby("question").agg({
        "is_correct": "mean",  # accuracy
        "input_tokens": "mean",
        "output_tokens": "mean", 
        "total_tokens": "mean",
        "total_cost": "mean",
        "retrieval_time": "mean",
        "llm_time": "mean",
        "total_time": "mean",
        "accuracy_per_1k_tokens": "mean"
    }).reset_index()
    
    question_metrics = question_metrics.rename(columns={
        "is_correct": "accuracy"
    })
    
    # Sort by accuracy
    question_metrics = question_metrics.sort_values("accuracy", ascending=False)
    
    return question_metrics

def display_cost_summary(df):
    """Display summary of cost and token usage"""
    # Overall summary
    total_queries = len(df)
    correct_queries = df["is_correct"].sum()
    incorrect_queries = total_queries - correct_queries
    accuracy = correct_queries / total_queries if total_queries > 0 else 0
    
    total_tokens = df["total_tokens"].sum()
    total_input_tokens = df["input_tokens"].sum()
    total_output_tokens = df["output_tokens"].sum()
    
    total_cost = df["total_cost"].sum()
    input_cost = df["input_cost"].sum()
    output_cost = df["output_cost"].sum()
    
    avg_tokens_per_query = total_tokens / total_queries if total_queries > 0 else 0
    avg_cost_per_query = total_cost / total_queries if total_queries > 0 else 0
    cost_per_correct_answer = total_cost / correct_queries if correct_queries > 0 else 0
    
    avg_accuracy_per_1k_tokens = 1000 * accuracy / avg_tokens_per_query if avg_tokens_per_query > 0 else 0
    
    # Display summary
    print("\n===== Cost and Token Usage Summary =====")
    
    summary_data = [
        ["Metric", "Value"],
        ["Total queries", f"{total_queries}"],
        ["Correct answers", f"{correct_queries} ({accuracy:.2%})"],
        ["Total tokens used", f"{total_tokens:,.0f}"],
        ["- Input tokens", f"{total_input_tokens:,.0f} ({total_input_tokens/total_tokens:.1%})"],
        ["- Output tokens", f"{total_output_tokens:,.0f} ({total_output_tokens/total_tokens:.1%})"],
        ["Total cost", f"${total_cost:.4f}"],
        ["- Input cost", f"${input_cost:.4f} ({input_cost/total_cost:.1%})"],
        ["- Output cost", f"${output_cost:.4f} ({output_cost/total_cost:.1%})"],
        ["Average tokens per query", f"{avg_tokens_per_query:.1f}"],
        ["Average cost per query", f"${avg_cost_per_query:.4f}"],
        ["Cost per correct answer", f"${cost_per_correct_answer:.4f}"],
        ["Accuracy per 1K tokens", f"{avg_accuracy_per_1k_tokens:.4f}"]
    ]
    
    print(tabulate(summary_data, headers="firstrow", tablefmt="grid"))
    
    # Save to CSV
    summary_df = pd.DataFrame({
        "Metric": [row[0] for row in summary_data[1:]],
        "Value": [row[1] for row in summary_data[1:]]
    })
    
    summary_df.to_csv(OUTPUT_DIR / "cost_summary.csv", index=False)
    print(f"Cost summary saved to {OUTPUT_DIR / 'cost_summary.csv'}")
    
    return {
        "total_queries": total_queries,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "avg_tokens_per_query": avg_tokens_per_query,
        "cost_per_correct_answer": cost_per_correct_answer,
        "avg_accuracy_per_1k_tokens": avg_accuracy_per_1k_tokens
    }

def display_latency_summary(df):
    """Display summary of latency metrics"""
    # Fill any missing timing values with 0
    for col in ["retrieval_time", "llm_time", "total_time"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0
    
    # Calculate averages
    avg_retrieval_time = df["retrieval_time"].mean()
    avg_llm_time = df["llm_time"].mean()
    avg_total_time = df["total_time"].mean()
    
    # Display summary
    print("\n===== Latency Summary =====")
    
    latency_data = [
        ["Metric", "Average (seconds)"],
        ["Total response time", f"{avg_total_time:.2f}"],
        ["Retrieval time", f"{avg_retrieval_time:.2f} ({avg_retrieval_time/avg_total_time:.1%})"],
        ["LLM inference time", f"{avg_llm_time:.2f} ({avg_llm_time/avg_total_time:.1%})"]
    ]
    
    print(tabulate(latency_data, headers="firstrow", tablefmt="grid"))
    
    # Save to CSV
    latency_df = pd.DataFrame({
        "Metric": [row[0] for row in latency_data[1:]],
        "Average": [float(row[1].split()[0]) for row in latency_data[1:]]
    })
    
    latency_df.to_csv(OUTPUT_DIR / "latency_summary.csv", index=False)
    print(f"Latency summary saved to {OUTPUT_DIR / 'latency_summary.csv'}")
    
    return {
        "avg_retrieval_time": avg_retrieval_time,
        "avg_llm_time": avg_llm_time,
        "avg_total_time": avg_total_time
    }

def display_question_efficiency(question_metrics):
    """Display efficiency metrics by question type"""
    # Select columns to display
    display_cols = [
        "question", 
        "accuracy", 
        "total_tokens", 
        "total_cost", 
        "total_time",
        "accuracy_per_1k_tokens"
    ]
    
    # Prepare data for display
    display_df = question_metrics[display_cols].copy()
    display_df["accuracy"] = display_df["accuracy"].apply(lambda x: f"{x:.2%}")
    display_df["total_tokens"] = display_df["total_tokens"].apply(lambda x: f"{x:.1f}")
    display_df["total_cost"] = display_df["total_cost"].apply(lambda x: f"${x:.4f}")
    display_df["total_time"] = display_df["total_time"].apply(lambda x: f"{x:.2f}s")
    display_df["accuracy_per_1k_tokens"] = display_df["accuracy_per_1k_tokens"].apply(lambda x: f"{x:.4f}")
    
    # Rename columns for display
    display_df.columns = [
        "Question", 
        "Accuracy", 
        "Avg. Tokens", 
        "Avg. Cost", 
        "Avg. Time",
        "Accuracy per 1K tokens"
    ]
    
    # Shorten question text for display
    display_df["Question"] = display_df["Question"].apply(
        lambda x: x[:45] + "..." if len(x) > 45 else x
    )
    
    # Display table
    print("\n===== Efficiency Metrics by Question Type =====")
    print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=False))
    
    # Save to CSV
    question_metrics.to_csv(OUTPUT_DIR / "question_efficiency.csv", index=False)
    print(f"Question efficiency metrics saved to {OUTPUT_DIR / 'question_efficiency.csv'}")

def create_visualizations(df, question_metrics, cost_summary, latency_summary):
    """Create visualizations for cost and latency metrics"""
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Cost vs. Accuracy by question type
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="total_cost", 
        y="accuracy", 
        size="total_tokens",
        hue="accuracy_per_1k_tokens",
        data=question_metrics,
        sizes=(50, 300),
        palette="viridis",
        alpha=0.7
    )
    
    # Add labels to points
    for i, row in question_metrics.iterrows():
        question_text = row["question"]
        short_text = question_text[:30] + "..." if len(question_text) > 30 else question_text
        plt.annotate(
            short_text, 
            (row["total_cost"], row["accuracy"]),
            fontsize=8,
            alpha=0.8,
            ha='center',
            va='bottom',
            xytext=(0, 5),
            textcoords='offset points'
        )
    
    plt.title("Cost vs. Accuracy by Question Type")
    plt.xlabel("Average Cost per Query ($)")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cost_vs_accuracy.png")
    
    # 2. Token distribution (input vs. output)
    plt.figure(figsize=(8, 6))
    token_data = pd.DataFrame({
        'Type': ['Input Tokens', 'Output Tokens'],
        'Count': [df["input_tokens"].mean(), df["output_tokens"].mean()]
    })
    
    ax = sns.barplot(x='Type', y='Count', data=token_data, palette="Blues_d")
    plt.title("Average Token Distribution per Query")
    plt.ylabel("Tokens")
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.1f}', 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            fontsize=10,
            xytext=(0, 5),
            textcoords='offset points'
        )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "token_distribution.png")
    
    # 3. Latency breakdown
    plt.figure(figsize=(8, 6))
    latency_data = pd.DataFrame({
        'Component': ['Retrieval', 'LLM Inference', 'Other Processing'],
        'Time (s)': [
            latency_summary["avg_retrieval_time"], 
            latency_summary["avg_llm_time"],
            latency_summary["avg_total_time"] - latency_summary["avg_retrieval_time"] - latency_summary["avg_llm_time"]
        ]
    })
    
    ax = sns.barplot(x='Component', y='Time (s)', data=latency_data, palette="Greens_d")
    plt.title("Average Latency Breakdown per Query")
    plt.ylabel("Time (seconds)")
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.2f}s', 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            fontsize=10,
            xytext=(0, 5),
            textcoords='offset points'
        )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "latency_breakdown.png")
    
    # 4. Accuracy vs. Token Efficiency
    plt.figure(figsize=(10, 6))
    plt.scatter(
        question_metrics["accuracy"], 
        question_metrics["accuracy_per_1k_tokens"],
        s=question_metrics["total_tokens"] / 10,
        alpha=0.7,
        c=question_metrics["total_cost"],
        cmap="viridis"
    )
    
    plt.colorbar(label="Average Cost per Query ($)")
    plt.title("Accuracy vs. Token Efficiency by Question Type")
    plt.xlabel("Accuracy")
    plt.ylabel("Accuracy per 1K Tokens")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_vs_efficiency.png")
    
    print(f"\nVisualizations saved to {OUTPUT_DIR}")

def generate_recommendations(df, question_metrics, cost_summary):
    """Generate recommendations based on the cost and efficiency analysis"""
    # Sort questions by different metrics for analysis
    least_efficient = question_metrics.sort_values("accuracy_per_1k_tokens").iloc[0]
    most_efficient = question_metrics.sort_values("accuracy_per_1k_tokens", ascending=False).iloc[0]
    most_expensive = question_metrics.sort_values("total_cost", ascending=False).iloc[0]
    highest_accuracy = question_metrics.sort_values("accuracy", ascending=False).iloc[0]
    lowest_accuracy = question_metrics.sort_values("accuracy").iloc[0]
    
    # Generate recommendations
    print("\n===== Cost Optimization Recommendations =====")
    
    recommendations = []
    
    # 1. Token efficiency recommendations
    recommendations.append("Token Efficiency Improvements:")
    
    if cost_summary["avg_accuracy_per_1k_tokens"] < 0.5:
        recommendations.append(f"- Overall token efficiency is low ({cost_summary['avg_accuracy_per_1k_tokens']:.4f} accuracy per 1K tokens)")
        recommendations.append("  Consider optimizing prompts to reduce token usage while maintaining accuracy")
    
    recommendations.append(f"- Most token-efficient question: '{most_efficient['question']}'")
    recommendations.append(f"  Efficiency: {most_efficient['accuracy_per_1k_tokens']:.4f} accuracy per 1K tokens")
    recommendations.append(f"- Least token-efficient question: '{least_efficient['question']}'")
    recommendations.append(f"  Efficiency: {least_efficient['accuracy_per_1k_tokens']:.4f} accuracy per 1K tokens")
    recommendations.append(f"  Consider revising the retrieval or prompt approach for this question type")
    
    # 2. Cost optimization recommendations
    recommendations.append("\nCost Optimization Strategies:")
    
    # Calculate input/output token ratio
    input_ratio = df["input_tokens"].mean() / df["total_tokens"].mean() if df["total_tokens"].mean() > 0 else 0
    
    if input_ratio > 0.8:
        recommendations.append("- Input tokens account for a high percentage of token usage")
        recommendations.append("  Consider:")
        recommendations.append("  1. Reducing context size by improving chunk relevance")
        recommendations.append("  2. Using more efficient chunking strategies")
        recommendations.append("  3. Implementing a filtering step to remove less relevant chunks")
    
    recommendations.append(f"- Most expensive question type: '{most_expensive['question']}'")
    recommendations.append(f"  Average cost: ${most_expensive['total_cost']:.4f} per query")
    
    if most_expensive['accuracy'] < 0.8:
        recommendations.append(f"  Note: This question type also has relatively low accuracy ({most_expensive['accuracy']:.2%})")
        recommendations.append("  Focus optimization efforts here for maximum cost-benefit impact")
    
    # 3. Performance optimization recommendations
    recommendations.append("\nPerformance Optimization Opportunities:")
    
    if lowest_accuracy["question"] == least_efficient["question"]:
        recommendations.append(f"- Critical improvement needed for: '{lowest_accuracy['question']}'")
        recommendations.append(f"  This question type has both low accuracy ({lowest_accuracy['accuracy']:.2%}) and low token efficiency")
    else:
        recommendations.append(f"- Focus on improving: '{lowest_accuracy['question']}'")
        recommendations.append(f"  Current accuracy: {lowest_accuracy['accuracy']:.2%}")
    
    # Calculate potential savings
    current_cost_per_correct = cost_summary["cost_per_correct_answer"]
    if cost_summary["accuracy"] < 0.9:
        target_accuracy = min(0.9, cost_summary["accuracy"] * 1.2)  # 20% improvement or 90%, whichever is lower
        potential_cost_per_correct = cost_summary["total_cost"] / (target_accuracy * cost_summary["total_queries"])
        potential_savings = current_cost_per_correct - potential_cost_per_correct
        
        recommendations.append(f"\nPotential Impact:")
        recommendations.append(f"- Current cost per correct answer: ${current_cost_per_correct:.4f}")
        recommendations.append(f"- By improving accuracy to {target_accuracy:.2%}, cost could reduce to ${potential_cost_per_correct:.4f} per correct answer")
        recommendations.append(f"- Potential savings: ${potential_savings:.4f} per correct answer ({potential_savings/current_cost_per_correct:.1%} reduction)")
    
    # Print recommendations
    for rec in recommendations:
        print(rec)
    
    # Save recommendations to file
    with open(OUTPUT_DIR / "cost_optimization_recommendations.txt", "w") as f:
        f.write("\n".join(recommendations))
    
    print(f"\nRecommendations saved to {OUTPUT_DIR / 'cost_optimization_recommendations.txt'}")

def main():
    """Main function to run the cost and latency analysis"""
    try:
        # Load data
        df, evaluation = load_data()
        
        # Calculate cost metrics
        df = calculate_cost_metrics(df)
        
        # Group by question type
        question_metrics = analyze_metrics_by_question(df)
        
        # Display summaries
        cost_summary = display_cost_summary(df)
        latency_summary = display_latency_summary(df)
        display_question_efficiency(question_metrics)
        
        # Create visualizations
        create_visualizations(df, question_metrics, cost_summary, latency_summary)
        
        # Generate recommendations
        generate_recommendations(df, question_metrics, cost_summary)
        
        print("\nCost and latency analysis completed successfully.")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required file - {e}")
    except json.JSONDecodeError:
        print(f"Error: One of the input files contains invalid JSON")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")

if __name__ == "__main__":
    main()