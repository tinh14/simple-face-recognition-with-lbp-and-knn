import cv2
import numpy as np
import os
import constants as c
import local_binary_pattern as lbp

features = []
labels = []

for person_name in os.listdir(c.PROCESSED_DATASET_PATH):
	person_dir = os.path.join(c.PROCESSED_DATASET_PATH, person_name)
	for img_name in os.listdir(person_dir):
		img_path = os.path.join(person_dir, img_name)
		
		img_bgr = cv2.imread(img_path, cv2.COLOR_BGRA2BGR)

		img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.resize(img_gray, (c.IMAGE_SIZE, c.IMAGE_SIZE))

		img_lbp = lbp.LBP(img_gray, img_gray.shape)
		
		vector_lbp = img_lbp.flatten()

		features.append(vector_lbp)
		labels.append(person_name)

		print(img_path)
		

features = np.array(features)
labels = np.array(labels)

np.save(c.FEATURES_PATH, features);
np.save(c.LABELS_PATH, labels);