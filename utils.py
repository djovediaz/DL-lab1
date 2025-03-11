import matplotlib.pyplot as plt
import numpy as np

def plot(history):
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(accuracy) + 1)

    plt.plot(epochs, accuracy, "b-o", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "r--o", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, "b-o", label="Training loss")
    plt.plot(epochs, val_loss, "r-o", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()

def plot_label_distribution(y, title):
    labels, counts = np.unique(y, return_counts=True)
    labels = labels.astype(int)  # Ensure labels are integers

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts, align='center', alpha=0.7)
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.title(title)
    plt.xticks(labels)  # Ensure x-axis shows integer labels
    plt.show()