import torch
import matplotlib.pyplot as plt
import seaborn as sns
from mqar_minimal import MQARModel, generate_mqar

def plot_lambda():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = 256
    model = MQARModel(vocab_size=8192, d_model=64, num_heads=1, 
                      state_size=64, num_levels=9).to(device)
    
    # Load your best checkpoint if you have one, else this shows initial state
    x, y = generate_mqar(1, seq_len, 4, vocab_size=8192)
    x = x.to(device)
    
    # Hook to capture lambda (L) from the first layer
    lambdas = []
    def hook(module, input, output):
        # In your HAttentionLayer, 'L' is what we want to see
        # We need to capture it during the forward pass
        pass 

    # For a quick check, let's just grab the projection weights
    layer = model.layers[0]
    with torch.no_grad():
        # Simulate the dl path
        dl = layer.dl_proj(model.embedding(x)) # [B, T, H*L]
        B, T, _ = dl.shape
        dl = dl.view(B, T, layer.num_heads, layer.num_levels)
        
        if layer.lambda_mode == "fixed":
            L = torch.nn.functional.softplus(layer.L[None, None, :, :] * dl)
        else:
            L = layer.lambda_module(dl)
            
    # Plotting the first head of the first sample
    plt.figure(figsize=(12, 6))
    sns.heatmap(L[0, :, 0, :].cpu().numpy().T, cmap="viridis")
    plt.title(f"Lambda Weights across Sequence (Mode: {layer.lambda_mode})")
    plt.xlabel("Sequence Position")
    plt.ylabel("Memory Level (l)")
    plt.savefig("lambda_heatmap.png")
    print("Saved heatmap to lambda_heatmap.png")

if __name__ == "__main__":
    plot_lambda()
