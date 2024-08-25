import matplotlib.pyplot as plt

epochs = []
train_losses = []
val_losses = []
train_F1_scores = []
val_F1_scores = []
train_precisions = []
val_precisions = []
train_recalls = []
val_recalls = []
train_aucs = []
val_aucs = []
train_prs = []
val_prs = []


with open('./log_file_20240824-212043_.txt', 'r') as f:
    for i, line in enumerate(f):
        parts = line.strip().split(', ')
        epochs.append(i+1)
        train_losses.append(float(parts[1].split(':')[1]))
        val_losses.append(float(parts[2].split(':')[1]))
        train_F1_scores.append(float(parts[3].split(':')[1]))
        val_F1_scores.append(float(parts[4].split(':')[1]))
        train_precisions.append(float(parts[5].split(':')[1]))
        val_precisions.append(float(parts[6].split(':')[1]))
        train_recalls.append(float(parts[7].split(':')[1]))
        val_recalls.append(float(parts[8].split(':')[1]))
        train_aucs.append(float(parts[9].split(':')[1]))
        val_aucs.append(float(parts[10].split(':')[1]))
        train_prs.append(float(parts[11].split(':')[1]))
        val_prs.append(float(parts[12].split(':')[1]))
print(epochs)
print(train_losses)
print(val_aucs)
def plot_metric(metric_name, train_values, val_values):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, label=f'Train {metric_name}')
    plt.plot(epochs, val_values, label=f'Val {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Over Epochs')
    plt.legend()
    plt.ylim([0.8,1.1])
    plt.grid(True)
    plt.savefig(f'pictures/{metric_name.lower()}_over_epochs.png')
    plt.show()


plot_metric('Loss', train_losses, val_losses)
plot_metric('F1 Score', train_F1_scores, val_F1_scores)
plot_metric('Precision', train_precisions, val_precisions)
plot_metric('Recall', train_recalls, val_recalls)
plot_metric('AUC', train_aucs, val_aucs)
plot_metric('PR', train_prs, val_prs)
