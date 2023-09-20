import cv2
import os
import shutil
import constants as c
import matplotlib.pyplot as plt

def processing(img_bgr):
	face_classifier = cv2.CascadeClassifier(
	    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
	)
	img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(
	    img_gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100)
	)
	face_img = None
	if len(faces) > 0:
	    first_face = faces[0]
	    x, y, w, h = first_face
	    face_img = img_bgr[y:y+h, x:x+w]
	return face_img;

def init():
	# remove all dics and files in processed data set
	if os.path.exists(c.PROCESSED_DATASET_PATH):
		shutil.rmtree(c.PROCESSED_DATASET_PATH)
		os.mkdir(c.PROCESSED_DATASET_PATH)

	for person_name in os.listdir(c.DATASET_PATH):
		
		person_dir = os.path.join(c.DATASET_PATH, person_name)

		os.makedirs(os.path.join(c.PROCESSED_DATASET_PATH, person_name))

		i = 0
		for img_name in os.listdir(person_dir):
			img_path = os.path.join(person_dir, img_name)

			img_bgr = cv2.imread(img_path, cv2.COLOR_BGRA2BGR)

			face_image = processing(img_bgr)

			path = f"{c.PROCESSED_DATASET_PATH}/{person_name}/{i}.jpg"

			if (face_image is not None):
				cv2.imwrite(path, face_image)
				i = i + 1
				print(img_path)

if __name__ == "__main__":
    init()