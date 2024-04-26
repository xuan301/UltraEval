def parse_sparsity_log(log_content):
    original_sparsity = []
    llmpruning_sparsity = []
    act_sparsity = []
    pred_x_sparsity_before = []

    for line in log_content.splitlines():
        if "sparsity originally" in line:
            parts = line.split(':')
            original_sparsity.append(float(parts[1].strip()))
        elif "sparsity with LLMPruning" in line:
            parts = line.split(':')
            llmpruning_sparsity.append(float(parts[1].strip()))
        elif "sparsity after activation" in line:
            parts = line.split(':')
            act_sparsity.append(float(parts[1].strip()))
        elif "predicted sparsity" in line:
            parts = line.split(':')
            pred_x_sparsity_before.append(float(parts[1].strip()))
    return original_sparsity, llmpruning_sparsity, act_sparsity, pred_x_sparsity_before

log_file_path = 'LOG_SPARSITY'
with open(log_file_path, 'r') as file:
    log_content = file.read()

original_sparsity, llmpruning_sparsity, act_sparsity, pred_x_sparsity_before = parse_sparsity_log(log_content)

average_original_sparsity = sum(original_sparsity) / len(original_sparsity) if original_sparsity else -1 
average_llmpruning_sparsity = sum(llmpruning_sparsity) / len(llmpruning_sparsity) if llmpruning_sparsity else -1
average_act_sparsity = sum(act_sparsity) / len(act_sparsity) if act_sparsity else -1
average_pred_x_sparsity_before = sum(pred_x_sparsity_before) / len(pred_x_sparsity_before) if pred_x_sparsity_before else -1

print(f"Average original sparsity (before act): {average_original_sparsity}")
print(f"Average LLMPruning sparsity (before act): {average_llmpruning_sparsity}")
print(f"Average activation sparsity (after act): {average_act_sparsity}")
print(f"Average predicted sparsity (before act): {average_pred_x_sparsity_before}")