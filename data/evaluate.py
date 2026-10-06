"""evaluate.py - Evaluation script
This script evaluates the predictions made by a model against the gold standard labels. It calculates macro-F1 for each task (sentiment, sarcasm) and each dialect (AU, UK) separately, producing four component scores: `sent-AU/UK` and `sarc-AU/UK`. To reward robustness across varieties, each task is scored by its weakest-performing variety: 

$$
\text{score} = \frac{\min(\text{sent-AU, sent-UK}) + \min(\text{sarc-AU, sarc-UK})}{2}
$$
"""

import pandas as pd
from sklearn.metrics import f1_score

def evaluate(run_filepath, gold_filepath):
    # Load the gold standard labels
    gold_df = pd.read_csv(gold_filepath)

    # Load the model predictions
    run_df = pd.read_csv(run_filepath)

    # Collect the task /dialect f1 scores. The CSV files have columns like 'source', 'variety', 'text', 'sentiment', 'sarcasm'. We will use the 'variety' column to separate AU and UK, and the 'sentiment' and 'sarcasm' columns for the tasks.
    scores = {}
    dialects = ['en-AU', 'en-UK']
    for task in ['sentiment', 'sarcasm']:
        for dialect in dialects:
            # Filter the gold and run dataframes for the current task and dialect
            gold_filtered = gold_df[gold_df['variety'] == dialect][task]
            run_filtered = run_df[run_df['variety'] == dialect][task]

            # Calculate macro-F1 score
            f1 = f1_score(gold_filtered, run_filtered, average='macro')
            print(f"F1 score for {task} ({dialect}): {f1:.4f}")
            scores[f"{task}-{dialect}"] = f1

    # Calculate the final score as the average of the weakest-performing varieties for each task
    sent_scores = [scores[f"sentiment-{dialect}"] for dialect in dialects]
    sarc_scores = [scores[f"sarcasm-{dialect}"] for dialect in dialects]
    final_score = (min(sent_scores) + min(sarc_scores)) / 2
    print(f"Final score: {final_score:.4f}")

    # Return the scores dictionary and the final score
    return scores, final_score

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <run_filepath> <gold_filepath>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)
        sys.exit(1)

    if not os.path.isdir(truth_dir):
        print("%s doesn't exist" % truth_dir)
        sys.exit(1)

    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    

    run_filepath = os.path.join(submit_dir, "answer.csv")
    gold_filepath = os.path.join(truth_dir, "truth.csv")
    output_filepath = os.path.join(output_dir, 'scores.txt')

    scores, final_score = evaluate(run_filepath, gold_filepath)

    print("\nComponent scores:")
    for key, value in scores.items():
        print(f"f1-{key}: {value:.4f}")
    print(f"\nFinal score: {final_score:.4f}")

    # Write the scores to the output file
    with open(output_filepath, 'w') as f:
        for key, value in scores.items():
            f.write(f"f1-{key}: {value:.4f}\n")
        f.write(f"score: {final_score:.4f}\n")
