"""
Straightforward implementation of the ROUGE (Recall-Oriented
Understudy for Gisting Evaluation) metric. It calculates ROUGE-N (with n=1 and n=2) and ROUGE-L,
offering insights into the lexical overlap between a candidate summary and a
reference document.

The script is designed to be self-contained, with no external dependencies
beyond standard Python libraries, making it easy to integrate into various
projects.
"""

from collections import Counter
import re

def get_ngrams(text, n):
    """
    Calculates n-grams for a given text.

    Args:
        text (str): The input text.
        n (int): The order of n-grams to generate.

    Returns:
        Counter: A Counter object mapping each n-gram to its frequency.
    """
    words = re.split(r'\s+', text.lower())
    ngrams = Counter()
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngrams[ngram] += 1
    return ngrams

def calculate_rouge_n(candidate, reference, n):
    """
    Calculates the ROUGE-N score.

    Args:
        candidate (str): The candidate summary.
        reference (str): The reference document.
        n (int): The order of n-grams (e.g., 1 for ROUGE-1, 2 for ROUGE-2).

    Returns:
        dict: A dictionary containing the precision, recall, and F1-score.
    """
    candidate_ngrams = get_ngrams(candidate, n)
    reference_ngrams = get_ngrams(reference, n)

    if not candidate_ngrams or not reference_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    overlapping_ngrams = candidate_ngrams & reference_ngrams
    overlapping_count = sum(overlapping_ngrams.values())

    candidate_total = sum(candidate_ngrams.values())
    reference_total = sum(reference_ngrams.values())

    precision = overlapping_count / candidate_total if candidate_total > 0 else 0.0
    recall = overlapping_count / reference_total if reference_total > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def get_lcs(candidate, reference):
    """
    Finds the longest common subsequence between two texts.

    Args:
        candidate (str): The candidate summary.
        reference (str): The reference document.

    Returns:
        int: The length of the longest common subsequence.
    """
    candidate_words = re.split(r'\s+', candidate.lower())
    reference_words = re.split(r'\s+', reference.lower())

    m, n = len(candidate_words), len(reference_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if candidate_words[i-1] == reference_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

def calculate_rouge_l(candidate, reference):
    """
    Calculates the ROUGE-L score.

    Args:
        candidate (str): The candidate summary.
        reference (str): The reference document.

    Returns:
        dict: A dictionary containing the precision, recall, and F1-score for ROUGE-L.
    """
    lcs_length = get_lcs(candidate, reference)
    candidate_words = re.split(r'\s+', candidate.lower())
    reference_words = re.split(r'\s+', reference.lower())

    m, n = len(candidate_words), len(reference_words)

    if m == 0 or n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    precision = lcs_length / m
    recall = lcs_length / n
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def evaluate_rouge(candidate_summary, reference_document):
    """
    Performs a comprehensive ROUGE evaluation.

    Args:
        candidate_summary (str): The summary to be evaluated.
        reference_document (str): The reference document.

    Returns:
        dict: A dictionary containing the scores for ROUGE-1, ROUGE-2, and ROUGE-L.
    """
    return {
        "rouge_1": calculate_rouge_n(candidate_summary, reference_document, 1),
        "rouge_2": calculate_rouge_n(candidate_summary, reference_document, 2),
        "rouge_l": calculate_rouge_l(candidate_summary, reference_document),
    }

if __name__ == '__main__':
    # Example Usage
    candidate_summary = "The cat sat on the mat."
    reference_document = "A cat was sitting on the mat."

    print("Candidate Summary:", candidate_summary)
    print("Reference Document:", reference_document)

    scores = evaluate_rouge(candidate_summary, reference_document)

    print("\nROUGE Scores:")
    for rouge_type, results in scores.items():
        print(f"  {rouge_type.upper()}:")
        for metric, value in results.items():
            print(f"    {metric.capitalize()}: {value:.4f}")
