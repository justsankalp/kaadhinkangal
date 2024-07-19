# # # import pickle
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score
# # import numpy as np
# # from sklearn.preprocessing import LabelEncoder

# # data_dict = pickle.load(open('./data.pickle', 'rb'))

# # data = np.asarray(data_dict['data'])
# # labels = np.asarray(data_dict['labels'])

# # label_encoder = LabelEncoder()
# # labels_encoded = label_encoder.fit_transform(labels)

# # x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

# # li = label_encoder.inverse_transform(list(range(31)))
# # print(set(data_dict['labels']))
# # for i in range(31):
# #     print(f'{labels[i*900]} is mapped to {label_encoder.inverse_transform(list(range(31)))}')
# # model = RandomForestClassifier()
# # model.fit(x_train, y_train)

# # y_predict = model.predict(x_test)
# # score = accuracy_score(y_predict, y_test)

# # print('{}% of samples were classified correctly !'.format(score * 100))

# # # Save the model and label encoder
# # with open('model.p', 'wb') as f:
# #     pickle.dump({'model': model, 'label_encoder': label_encoder}, f)
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
print(set(labels))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.7, shuffle=True, stratify=labels_encoded)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate the accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Calculate additional metrics for each class
f1_per_class = f1_score(y_test, y_predict, average=None)
recall_per_class = recall_score(y_test, y_predict, average=None)
precision_per_class = precision_score(y_test, y_predict, average=None)
accuracy_per_class = []

# Calculate per class accuracy
conf_matrix = confusion_matrix(y_test, y_predict)
for i in range(len(label_encoder.classes_)):
    accuracy_per_class.append(conf_matrix[i, i] / conf_matrix[i].sum())

# Plot the metrics for each class
classes = label_encoder.classes_

plt.figure(figsize=(12, 6))
plt.plot(classes, f1_per_class, marker='o', label='F1 Score')
plt.plot(classes, recall_per_class, marker='o', label='Recall')
plt.plot(classes, precision_per_class, marker='o', label='Precision')
plt.plot(classes, accuracy_per_class, marker='o', label='Accuracy')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Metrics for Each Class')
plt.legend()
plt.grid(True)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Print overall metrics
print(f"Accuracy: {score * 100:.2f}%")
print(f"F1 Score (weighted): {f1_score(y_test, y_predict, average='weighted'):.2f}")
print(f"Recall (weighted): {recall_score(y_test, y_predict, average='weighted'):.2f}")
print(f"Precision (weighted): {precision_score(y_test, y_predict, average='weighted'):.2f}")

# Display the mapping from encoded labels to original labels
print("Label mapping:")
for i in range(len(label_encoder.classes_)):
    print(f"{i}: {label_encoder.inverse_transform([i])[0]}")

# Save the model and label encoder
with open('model_dummy_2.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)

print("Model and label encoder saved to model.p")

# print(f"Confusion Matix:{conf_matrix}")
# Calculate TP, FP, TN, FN
# if conf_matrix.shape == (2, 2):
#     tn, fp, fn, tp = conf_matrix.ravel()
#     print("Confusion Matrix:")
#     print(f"             Predicted Positive    Predicted Negative")
#     print(f"Actual Positive   TP: {tp}                FN: {fn}")
#     print(f"Actual Negative   FP: {fp}                TN: {tn}")
# else:
#     print("Confusion matrix is not 2x2. Unable to print TP, FP, TN, FN.")

# Display the mapping from encoded labels to original labels
# print("Label mapping:")
# for i in range(len(label_encoder.classes_)):
#     print(f"{i}: {label_encoder.inverse_transform([i])[0]}")

# # Save the model and label encoder
# with open('model_dummy_2.p', 'wb') as f:
#     pickle.dump({'model': model, 'label_encoder': label_encoder}, f)

# print("Model and label encoder saved to model.p")

# # import pickle
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score
# # import numpy as np
# # from sklearn.preprocessing import LabelEncoder


# # data_dict = pickle.load(open('./data.pickle', 'rb'))
# # data = np.asarray(data_dict['data'])
# # labels = np.asarray(data_dict['labels'])
# # print(set(labels))
# # # Separate the data for left and right hands
# # left_indices = [i for i, label in enumerate(labels) if 'Left' in label]
# # right_indices = [i for i, label in enumerate(labels) if 'Right' in label]
# # print(len(left_indices))
# # print(len(right_indices))
# # left_hand_data = data[left_indices]
# # labels_left = labels[left_indices]
# # right_hand_data = data[right_indices]
# # labels_right = labels[right_indices]
# # print(len(labels_left))
# # print(len(labels_right))
# # # Encode the labels for left hand
# # left_hand_label_encoder = LabelEncoder()
# # left_hand_labels_encoded = left_hand_label_encoder.fit_transform(labels_left)

# # # Encode the labels for right hand
# # right_hand_label_encoder = LabelEncoder()
# # right_hand_labels_encoded = right_hand_label_encoder.fit_transform(labels_right)

# # # Split the data into training and testing sets for left hand
# # x_left_train, x_left_test, y_left_train, y_left_test = train_test_split(
# #     left_hand_data, left_hand_labels_encoded, test_size=0.2, shuffle=True, stratify=left_hand_labels_encoded
# # )

# # # Split the data into training and testing sets for right hand
# # x_right_train, x_right_test, y_right_train, y_right_test = train_test_split(
# #     right_hand_data, right_hand_labels_encoded, test_size=0.2, shuffle=True, stratify=right_hand_labels_encoded
# # )

# # # Initialize and train the model for left hand
# # left_hand_model = RandomForestClassifier()
# # left_hand_model.fit(x_left_train, y_left_train)

# # # Initialize and train the model for right hand
# # right_hand_model = RandomForestClassifier()
# # right_hand_model.fit(x_right_train, y_right_train)

# # # Make predictions for left hand
# # y_left_predict = left_hand_model.predict(x_left_test)

# # # Calculate the accuracy for left hand
# # left_hand_score = accuracy_score(y_left_predict, y_left_test)
# # print('Left Hand - {}% of samples were classified correctly!'.format(left_hand_score * 100))

# # # Make predictions for right hand
# # y_right_predict = right_hand_model.predict(x_right_test)

# # # Calculate the accuracy for right hand
# # right_hand_score = accuracy_score(y_right_predict, y_right_test)
# # print('Right Hand - {}% of samples were classified correctly!'.format(right_hand_score * 100))

# # # Save the models and label encoders for both hands
# # with open('left_hand_model.p', 'wb') as f:
# #     pickle.dump({'model': left_hand_model, 'label_encoder': left_hand_label_encoder}, f)

# # with open('right_hand_model.p', 'wb') as f:
# #     pickle.dump({'model': right_hand_model, 'label_encoder': right_hand_label_encoder}, f)

# # print("Models and label encoders saved.")
