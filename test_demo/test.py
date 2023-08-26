
import pandas as pd
import numpy as np
all_predictions = [[0.1,0.9,0.2],[0.3,0.4,0.5],[0.6,0.7,0.8]]
all_labels = [[0,1,0],[0,0,1],[1,1,0]]
# Convert the list of predictions and labels to NumPy arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calculate the number of positive labels for each sample
num_pos_labels = np.sum(all_labels, axis=1)
print(num_pos_labels)
top_n_indices = np.argsort(all_predictions, axis=1)
print(top_n_indices)
# Get the indices of sorted predictions for each sample
sorted_indices = np.argsort(all_predictions, axis=1)[:, ::-1]
print(sorted_indices)
# Populate the DataFrame with the results
for i in range(len(top_n_indices)):
    true_label_positions = np.where(all_labels[i] == 1)[0]  # Get the positions of true labels
    num_pos_labels = len(true_label_positions)  # Count of true positive labels
    top_n_positions = sorted_indices[i, :num_pos_labels]  # Get the top n predicted positions



# num_pos_labels，对all_predictions进行排序，取出前num_pos_labels[i]个值
for i in range(len(num_pos_labels)):
    # 从大到小排序，取出前num_pos_labels[i]个值所做的原始位置

    top_n_indices = np.argsort(all_predictions[i], axis=1)[-num_pos_labels[i]:]
    print(top_n_indices)
top_n_indices = np.argsort(all_predictions, axis=1)[:, -num_pos_labels[0]:]
top_n_indices = np.argsort(all_predictions, axis=1)[:, -num_pos_labels[0]:]

print(top_n_indices)
# Create a DataFrame to store the results
results_df = pd.DataFrame(columns=["Top_N_Predicted_Positions", "True_Label_Positions"])

# Populate the DataFrame with the results
for i in range(len(top_n_indices)):
    top_n_positions = list(top_n_indices[i])  # Convert the index to a string
    true_label_positions = " ".join(map(str, np.where(all_labels[i] == 1)[0]))
    results_df.loc[i] = [top_n_positions, true_label_positions]