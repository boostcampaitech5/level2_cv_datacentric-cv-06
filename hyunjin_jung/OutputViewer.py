import cv2
import json
import numpy as np

with open("./output.csv", encoding="utf-8") as file:
    ann = json.load(file)
images = list(ann['images'].keys())
image_dir = './data/medical/img/test/'

img_idx = 0
ch = ' '
while ch != 'q':
    image_file = images[img_idx]
    image = cv2.imread(image_dir + image_file)

    words = ann['images'][image_file]['words']
    for word_key in words.keys():
        points = np.array(words[word_key]['points']).astype(np.int64)
        cv2.polylines(image, [points], True, (255, 0,0), 2)

    # resize
    ratio = image.shape[1] / image.shape[0]
    nh = 1000
    nw = nh * ratio
    image = cv2.resize(image, (int(nw), nh))

    cv2.imshow("image", image)
    cv2.waitKey(0)
    img_idx+=1