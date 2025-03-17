import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib




def extract_hog_features(image):
    image = cv2.resize(image, (64, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)
    return features.flatten()

data = []
labels = []

data_dir = "Database"

for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    
    if os.path.isdir(label_dir):
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is not None:
                hog_features = extract_hog_features(image)
                data.append(hog_features)
                labels.append(label)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

clf = svm.SVC(kernel='linear',max_iter=10000)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

model_filename = "svm_model_update_2.pkl"
joblib.dump(clf, model_filename)
