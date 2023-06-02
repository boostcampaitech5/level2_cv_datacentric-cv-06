import cv2
import os
import json
import numpy as np

"""
Mike: 이미지의 윗 부분(QR 코드 근처, 타이틀 부분)을 기존 사진으로 훈련했을 때
모델이 잘 인식하지 못 하는 것을 확인했습니다.

따라서 이를 집중적으로 tuning 할 수 있는 데이터를 인위적으로 생성하기로 했습니다.
이미지의 윗 부분 4개를 합쳐서 새로운 이미지를 만들었습니다.

글씨의 크기를 최대한 일정하게 유지하기 위해
가로가 더 긴 사진은 제외하고 대략 이미지의 크기와 비슷한 4000 x 3000 사이즈의 사진을 생성했습니다.
"""

NUM_COLLAGE = 50 # 일단 200/4
OUTPUT_SIZE = (4000, 3000)
IMAGE_DIR = "./img/img"

img_paths = []
words = []

with open("annotation.json", encoding="utf-8") as f:
    anno = json.load(f)

images_dict = anno["images"]
images = os.listdir(IMAGE_DIR)

count = 1
new_anno = {}

for i in range(0, 160, 4):
    words = {}
    word_idx = 1
    output_img = np.zeros([OUTPUT_SIZE[0], OUTPUT_SIZE[1], 3], dtype=np.int64)

    img1 = cv2.imread(os.path.join(IMAGE_DIR, images[i]))
    img2 = cv2.imread(os.path.join(IMAGE_DIR, images[i+1]))
    img3 = cv2.imread(os.path.join(IMAGE_DIR, images[i+2]))
    img4 = cv2.imread(os.path.join(IMAGE_DIR, images[i+3]))

    _, w, _ = img1.shape
    scale = OUTPUT_SIZE[1] / w
    h = 1000 / scale
    output_img[:1000, :, :] = cv2.resize(img1[:int(h), :, :], (3000, 1000))

    for word in images_dict[images[i]]["words"].values():
        if max(word["points"][2][1], word["points"][3][1]) <= h:
            points = word["points"]
            bw, bh = points[2][0] - points[0][0], points[2][1] - points[0][1]
            word["points"] = [[point[0] * scale, point[1] * scale] for point in word["points"]]
            words.update({str(word_idx).zfill(4): word})
            word_idx += 1

    _, w, _ = img2.shape
    scale = OUTPUT_SIZE[1] / w
    h = 1000 * w / OUTPUT_SIZE[1]
    output_img[1000:2000, :, :] = cv2.resize(img2[:int(h), :, :], (3000, 1000))

    for word in images_dict[images[i+1]]["words"].values():
        if max(word["points"][2][1], word["points"][3][1]) <= h:
            points = word["points"]
            bw, bh = points[2][0] - points[0][0], points[2][1] - points[0][1]
            word["points"] = [[point[0] * scale, point[1] * scale + 1000] for point in word["points"]]
            words.update({str(word_idx).zfill(4): word})
            word_idx += 1

    _, w, _ = img3.shape
    scale = OUTPUT_SIZE[1] / w
    h = 1000 * w / OUTPUT_SIZE[1]
    output_img[2000:3000, :, :] = cv2.resize(img3[:int(h), :, :], (3000, 1000))

    for word in images_dict[images[i+2]]["words"].values():
        if max(word["points"][2][1], word["points"][3][1]) <= h:
            points = word["points"]
            bw, bh = points[2][0] - points[0][0], points[2][1] - points[0][1]
            word["points"] = [[point[0] * scale, point[1] * scale + 2000] for point in word["points"]]
            words.update({str(word_idx).zfill(4): word})
            word_idx += 1

    _, w, _ = img4.shape
    scale = OUTPUT_SIZE[1] / w
    h = 1000 * w / OUTPUT_SIZE[1]
    output_img[3000:4000, :, :] = cv2.resize(img4[:int(h), :, :], (3000, 1000))

    for word in images_dict[images[i+3]]["words"].values():
        if max(word["points"][2][1], word["points"][3][1]) <= h:
            points = word["points"]
            bw, bh = points[2][0] - points[0][0], points[2][1] - points[0][1]
            word["points"] = [[point[0] * scale, point[1] * scale + 3000] for point in word["points"]]
            words.update({str(word_idx).zfill(4): word})
            word_idx += 1

    new_anno["collage{}.jpg".format(count)] = {
        "words": words,
        "chars": {},
        "img_w": 3000,
        "img_h": 4000,
        "tags": [],
        "relations": {},
        "annotation_log": {
            "worker": "worker",
            "timestamp": "2023-05-25",
            "tool_version": "",
            "source": None
        },
        "license_tag": {
            "usability": True,
            "public": False,
            "commercial": True,
            "type": None,
            "holder": "Upstage"
        }
    }
    #cv2.imwrite("collage{}.jpg".format(count), output_img)
    count += 1

new_anno = {"images": new_anno}

with open("collage_annotation.json", "w", encoding="utf-8") as f:
    json.dump(new_anno, f, ensure_ascii=False, indent=4)