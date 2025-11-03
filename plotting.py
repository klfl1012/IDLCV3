import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def model_training_csv_to_history_dict(df):
    # Initialize the dictionary to hold the results
    result = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    # Iterate through the DataFrame and append values to the result
    for _, row in df.iterrows():
        # Training row
        if not pd.isna(row['train_acc']):
            result['train_acc'].append(row['train_acc'])
            result['train_loss'].append(row['train_loss'])

        # Validation row
        if not pd.isna(row['val_acc']):
            result['val_acc'].append(row['val_acc'])
            result['val_loss'].append(row['val_loss'])

    return result

def plot_confusion_matrix(cm, class_names=None, normalize=False, x_label_rotation = 45):
    # 1. Initialize class names - in the format of Class# if not explicitly provided
    if class_names is None:
        class_names = [f'Class{i+1}' for i in range(cm.shape[0])]

    # 2. Plot confusion matrix
    cm_ = cm.astype('float')

    # 2.1 Normalize the cm if required
    if normalize:
        cm_ = cm_ / cm_.sum(axis=1, keepdims=True)

    # 2.2 Configure the plot
    plt.figure(figsize=(8,6))
    plt.imshow(cm_, interpolation='nearest')
    plt.title('Confusion Matrix' + (' (normalized)' if normalize else ''))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=x_label_rotation)
    plt.yticks(tick_marks, class_names)
    thresh = cm_.max() / 2.

    # 2.3 Annotate the cells with counts or normalized values
    for i in range(cm_.shape[0]):
        for j in range(cm_.shape[1]):
            txt = f"{cm_[i, j]:.2f}" if normalize else f"{int(cm_[i, j])}"
            plt.text(j, i, txt,
                     horizontalalignment="center",
                     color="white" if cm_[i, j] > thresh else "black")
                     
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def multiclass_roc_auc(y_true, y_score, class_names):
    # 1. Initialization
    auc_scores = []
    
    # 2. Configure the plot
    plt.figure(figsize=(8,6))
    colors = plt.cm.get_cmap('tab10', len(class_names))

    # 3. Compute ROC curve and AUC for each class
    for i in range(len(class_names)):
        # 3.1 Compute the ROC curve & AUC
        fpr, tpr, _ = roc_curve(y_true, y_score[:, i], pos_label=i)
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)

        # 3.2 Plot
        plt.plot(fpr, tpr, color=colors(i), label=f'Class {class_names[i]} (AUC = {auc_score:.3f})')
        
    # 4. Plot the diagonal line
    plt.plot([0, 1], [0, 1], linestyle='--')

    # 5. More plot configurations
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    return auc_scores, np.mean(auc_scores)

def plot_history(hist):
    # 1. Train loss
    plt.figure(figsize=(8,3))
    plt.plot(hist['train_loss'], label='Train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Train loss
    plt.figure(figsize=(8,3))
    plt.plot(hist['train_loss'], label='Train loss')
    plt.plot(hist['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Train and validation accuracy
    plt.figure(figsize=(8,3))
    plt.plot(hist['train_acc'], label='Train Accuracy')
    plt.plot(hist['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_accuracies(dfs, num_epochs=200, epoch_span=1, bar_width=0.4):
    # 1. Bin / Segment the data based on epoch_span
    aggregated_data = {}
    for model_index, df in enumerate(dfs):
        aggregated_train_acc = df.groupby(df['epoch'] // epoch_span)['train_acc'].mean()
        aggregated_val_acc = df.groupby(df['epoch'] // epoch_span)['val_acc'].mean()

        # Store aggregated data in a dictionary
        aggregated_data[f'Model {model_index + 1}'] = {
            'train_acc': aggregated_train_acc,
            'val_acc': aggregated_val_acc
        }

    # 2. Plotting
    indices = np.arange(len(aggregated_train_acc))

    plt.figure(figsize=(12, 6))
    for model_index, (model_name, data) in enumerate(aggregated_data.items()):
        plt.bar(indices + model_index * bar_width, data['train_acc'], width=bar_width, 
                label=f'{model_name} Training', alpha=0.7)
        
    plt.xticks(indices + (bar_width * (len(aggregated_data)-1)) / 2, [f'Epoch {min(epoch_span * (i+1),num_epochs)}' for i in range(len(aggregated_train_acc))])
    plt.title('Training Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    for model_index, (model_name, data) in enumerate(aggregated_data.items()):
        plt.bar(indices + model_index * bar_width, data['val_acc'], width=bar_width, 
                label=f'{model_name} Validation', alpha=0.7)
    
    plt.xticks(indices + (bar_width * (len(aggregated_data)-1)) / 2, [f'Epoch {min(epoch_span * (i+1),num_epochs)}' for i in range(len(aggregated_train_acc))])
    plt.title('Validation Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch') 
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Unifying the training and validation bar color to improve readability
    training_colors = plt.cm.Blues(np.linspace(0.3, 0.7, len(aggregated_data)))  # Blue tones
    validation_colors = plt.cm.Blues(np.linspace(0.7, 0.9, len(aggregated_data)))  # Lighter blue tones

    plt.figure(figsize=(12, 6))
    for model_index, (model_name, data) in enumerate(aggregated_data.items()):
        plt.bar(indices + model_index * bar_width + bar_width* 0.25, data['train_acc'], width=bar_width / 2, label=f'{model_name} - Training', color=training_colors[model_index], alpha=0.7)
        plt.bar(indices + model_index * bar_width + (bar_width / 2), data['val_acc'], width=bar_width / 2, label=f'{model_name} - Validation', color=validation_colors[model_index], alpha=0.7)

    plt.xticks(indices + (bar_width * len(aggregated_data)) / 2, [f'Epoch {min(epoch_span * (i+1),num_epochs)}' for i in range(len(aggregated_train_acc))])
    plt.title('Training and Validation Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()