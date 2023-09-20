from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np
import local_binary_pattern as lbp
import constants as c
import preprocessing as p

features = np.load(c.FEATURES_PATH)
labels = np.load(c.LABELS_PATH)

knn_model = KNeighborsClassifier(n_neighbors = c.K)
knn_model.fit(features, labels)

img_bgr = cv2.imread(c.TEST_IMAGE_PATH, cv2.COLOR_BGRA2BGR)
img_bgr = p.processing(img_bgr)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_gray = cv2.resize(img_gray, (c.IMAGE_SIZE, c.IMAGE_SIZE))
img_lbp = lbp.LBP(img_gray, img_gray.shape)
test_feature = img_lbp.flatten()

predicted_label_array = knn_model.predict([test_feature])
probability_maxtrix = knn_model.predict_proba([test_feature])

predicted_label = predicted_label_array[0];
probability = round(probability_maxtrix[0][0] * 100, 2);

print(f"Predicted label: {predicted_label}")
print(f"Probability: {probability}%")