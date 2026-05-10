import matplotlib.pyplot as plt
import seaborn as sns


class EDAVisualizer:
    def __init__(self, style="whitegrid"):
        sns.set_style(style)

    def plot_correlation_heatmap(self, corr_matrix):
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()
    
    def plot_distributions(self, df, features, target_col):
        """Plots boxplots to check for outliers and class separability."""
        n_features = len(features)
        fig, axes = plt.subplots(nrows=(n_features // 3) + 1, ncols=3, figsize=(15, 12))
        axes = axes.flatten()

        for i, col in enumerate(features):
            sns.boxplot(data=df, x=target_col, y=col, ax=axes[i])
            axes[i].set_title(f'{col} by Target')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()