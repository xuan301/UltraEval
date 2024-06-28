import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_distributions(gate_x, fused_gate_x, pred_gate_x, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    layers = gate_x.keys()
    
    for layer in layers:
        gate_x_values = np.concatenate([v.flatten() for v in gate_x[layer]])
        fused_gate_x_values = np.concatenate([v.flatten() for v in fused_gate_x[layer]])
        pred_gate_x_values = np.concatenate([v.flatten() for v in pred_gate_x[layer]])
        
        # 计算非零值的比例
        gate_x_nonzero_percent = np.count_nonzero(gate_x_values) / len(gate_x_values) * 100
        fused_gate_x_nonzero_percent = np.count_nonzero(fused_gate_x_values) / len(fused_gate_x_values) * 100
        pred_gate_x_nonzero_percent = np.count_nonzero(pred_gate_x_values) / len(pred_gate_x_values) * 100
        
        print(f'Layer {layer}:')
        print(f'gate_x non-zero percentage: {gate_x_nonzero_percent:.2f}%')
        print(f'fused_gate_x non-zero percentage: {fused_gate_x_nonzero_percent:.2f}%')
        print(f'pred_gate_x non-zero percentage: {pred_gate_x_nonzero_percent:.2f}%')
        
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.hist(gate_x_values, bins=100, density=True, alpha=0.6, color='g')
        plt.title(f'Distribution of gate_x values (Layer {layer})')
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.annotate(f'Non-zero: {gate_x_nonzero_percent:.2f}%', xy=(0.95, 0.95), xycoords='axes fraction', 
                     fontsize=12, ha='right', va='top')
        
        plt.subplot(1, 3, 2)
        plt.hist(fused_gate_x_values, bins=100, density=True, alpha=0.6, color='b')
        plt.title(f'Distribution of fused_gate_x values (Layer {layer})')
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.annotate(f'Non-zero: {fused_gate_x_nonzero_percent:.2f}%', xy=(0.95, 0.95), xycoords='axes fraction', 
                     fontsize=12, ha='right', va='top')
        
        plt.subplot(1, 3, 3)
        plt.hist(pred_gate_x_values, bins=100, density=True, alpha=0.6, color='r')
        plt.title(f'Distribution of pred_gate_x values (Layer {layer})')
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.annotate(f'Non-zero: {pred_gate_x_nonzero_percent:.2f}%', xy=(0.95, 0.95), xycoords='axes fraction', 
                     fontsize=12, ha='right', va='top')
        
        plt.savefig(os.path.join(output_dir, f'layer_{layer}.png'))
        plt.close()  # Close the figure to free memory

def main():
    total_sparsity_count = 1000  # Adjust according to your actual situation
    gate_x_file = f'gate_x_{total_sparsity_count}.pkl'
    fused_gate_x_file = f'fused_gate_x_{total_sparsity_count}.pkl'
    pred_gate_x_file = f'pred_gate_x_{total_sparsity_count}.pkl'
    output_dir = 'plots_100'  # Directory to save the images
    
    gate_x = load_data(gate_x_file)
    fused_gate_x = load_data(fused_gate_x_file)
    pred_gate_x = load_data(pred_gate_x_file)
    
    plot_distributions(gate_x, fused_gate_x, pred_gate_x, output_dir)

if __name__ == "__main__":
    main()
