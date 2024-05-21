import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_metric(models, values, metric_name):

    # Create the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(models, values, color='skyblue')

    # Add title and labels
    plt.title(metric_name)
    plt.xlabel('Model')
    plt.ylabel('Values')

    # Show the plot
    plt.show()