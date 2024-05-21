import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_metric(metrics, title_name):
    fig, axs = plt.subplots(len(metrics.keys()), 1, figsize=(5, 12))

    for i, (metric, model_scores_dict) in enumerate(metrics.items()):
        models = model_scores_dict.keys()
        scores = model_scores_dict.values()
        axs[i].bar(models, scores, color='skyblue')
        axs[i].set_title(metric)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(title_name, fontsize=16)
    # Display the combined plot
    plt.show()