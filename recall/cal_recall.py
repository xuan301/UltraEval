import csv

def extract_recall_from_file(filepath):
    """
    Extracts recall values from a given file.
    
    Args:
    filepath (str): Path to the log file.
    
    Returns:
    dict: A dictionary where keys are layer numbers and values are recall values.
    """
    recalls = {}
    with open(filepath, 'r') as file:
        for line in file:
            if ',' in line:
                try:
                    layer, recall = line.strip().split(',')
                    recalls[int(layer)] = float(recall)
                except ValueError:
                    # Ignore lines that cannot be parsed
                    continue
    return recalls

def main():
    log_mask_path = 'LOG_mask'
    log_mask_2_path = 'LOG_mask_2'
    log_pred_path = 'LOG_pred'
    output_csv_path = 'layer_recall_comparison.csv'

    # Extract recall values
    mask_recalls = extract_recall_from_file(log_mask_path)
    mask_2_recalls = extract_recall_from_file(log_mask_2_path)
    pred_recalls = extract_recall_from_file(log_pred_path)

    # Combine keys (layer numbers) from all dictionaries
    all_layers = set(mask_recalls.keys()).union(set(mask_2_recalls.keys())).union(set(pred_recalls.keys()))

    # Write results to CSV file
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Layer', 'LOG_mask Recall', 'LOG_mask_2 Recall', 'LOG_pred Recall'])
        
        for layer in sorted(all_layers):
            mask_recall = mask_recalls.get(layer, 'N/A')
            mask_2_recall = mask_2_recalls.get(layer, 'N/A')
            pred_recall = pred_recalls.get(layer, 'N/A')
            csvwriter.writerow([layer, mask_recall, mask_2_recall, pred_recall])

    print(f"Results written to {output_csv_path}")

if __name__ == "__main__":
    main()
