"""
This script calculates BERTScore for a given candidate summary and a reference
document. BERTScore evaluates the semantic similarity between two texts using
contextual embeddings from BERT models.

This implementation specifically uses Inverse Document Frequency (IDF) weighting,
which gives higher importance to more informative words (i.e., words that are
rare across a corpus) and less importance to common words.
"""

from bert_score import score
import torch

def evaluate_bertscore(candidate_summary, reference_document, use_idf=True):
    """
    Calculates BERTScore between a candidate and reference text.

    Args:
        candidate_summary (str): The generated summary to be evaluated.
        reference_document (str): The reference document.
        use_idf (bool): Whether to use IDF weighting. Defaults to True but falls back to False if NaN values occur.

    Returns:
        dict: A dictionary containing the precision, recall, and F1 score as floats.
              Returns error information if an error occurs.
    """
    import math
    
    try:
        # The score function takes lists of candidates and references
        candidates = [candidate_summary]
        references = [reference_document]

        # First try with IDF weighting
        try:
            precision, recall, f1 = score(
                candidates,
                references,
                lang="en",
                model_type="bert-base-uncased",
                idf=use_idf,
                verbose=False
            )
            
            # Convert tensors to float values
            precision_val = float(precision.item())
            recall_val = float(recall.item())
            f1_val = float(f1.item())
            
            # Check for NaN values - if found, retry without IDF
            if math.isnan(precision_val) or math.isnan(recall_val) or math.isnan(f1_val):
                if use_idf:
                    print("Warning: NaN values detected with IDF=True, retrying with IDF=False")
                    return evaluate_bertscore(candidate_summary, reference_document, use_idf=False)
                else:
                    # If we still get NaN without IDF, replace with 0.0
                    precision_val = 0.0 if math.isnan(precision_val) else precision_val
                    recall_val = 0.0 if math.isnan(recall_val) else recall_val
                    f1_val = 0.0 if math.isnan(f1_val) else f1_val
            
            return {
                "precision": precision_val,
                "recall": recall_val,
                "f1_score": f1_val,
                "idf_enabled": use_idf and not math.isnan(precision_val) and not math.isnan(recall_val) and not math.isnan(f1_val),
                "model_type": "bert-base-uncased"
            }
            
        except Exception as idf_error:
            if use_idf:
                print(f"Warning: Error with IDF=True ({idf_error}), retrying with IDF=False")
                return evaluate_bertscore(candidate_summary, reference_document, use_idf=False)
            else:
                raise idf_error
                
    except Exception as e:
        return {
            "error": f"An error occurred during BERTScore calculation: {e}",
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "idf_enabled": False,
            "model_type": "bert-base-uncased"
        }

if __name__ == '__main__':
    # --- Example 1: Using string variables ---
    print("--- Example 1: Evaluating with string variables ---")
    candidate_summary_str = "The cat sat on the mat."
    reference_document_str = "A feline was resting on the rug."

    print(f"Candidate Summary: \"{candidate_summary_str}\"")
    print(f"Reference Document: \"{reference_document_str}\"")

    result = evaluate_bertscore(candidate_summary_str, reference_document_str)

    if "error" not in result:
        print("\nBERTScore Results:")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1 Score:  {result['f1_score']:.4f}")
        print(f"  IDF Enabled: {result['idf_enabled']}")
        print(f"  Model: {result['model_type']}")
    else:
        print(f"\nError: {result['error']}")

    # --- Example 2: Reading from files ---
    print("\n" + "="*50 + "\n")
    print("--- Example 2: Evaluating with files ---")
    try:
        # Create dummy files for the example
        with open("candidate_summary.txt", "w") as f:
            f.write("A high-speed train connects the two major cities.")
        with open("reference_document.txt", "w") as f:
            f.write("The two metropolitan areas are linked by a bullet train.")

        # Read content from the files
        with open("candidate_summary.txt", "r") as f:
            candidate_from_file = f.read()
        with open("reference_document.txt", "r") as f:
            reference_from_file = f.read()

        print(f"Candidate from candidate_summary.txt: \"{candidate_from_file}\"")
        print(f"Reference from reference_document.txt: \"{reference_from_file}\"")

        result_file = evaluate_bertscore(candidate_from_file, reference_from_file)

        if "error" not in result_file:
            print("\nBERTScore Results from files:")
            print(f"  Precision: {result_file['precision']:.4f}")
            print(f"  Recall:    {result_file['recall']:.4f}")
            print(f"  F1 Score:  {result_file['f1_score']:.4f}")
            print(f"  IDF Enabled: {result_file['idf_enabled']}")
            print(f"  Model: {result_file['model_type']}")
        else:
            print(f"\nError: {result_file['error']}")

    except IOError as e:
        print(f"\nCould not perform file-based evaluation: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")