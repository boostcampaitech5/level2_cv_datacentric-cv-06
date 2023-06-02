import cv2
import json
import numpy as np

'''
viewer.py를 응용해서 모든 사진을 for loop으로 볼 수 있게 만들었습니다.
'''

# 경로 설정
img_idx = 1
font_size = 1
with open("./collage_annotation.json", encoding="utf-8") as file:
    ann = json.load(file)
#img_file = list(ann.get("images").keys())[img_idx]
#img = ann.get("images")[img_file]
#words = img["words"]

img_files = list(ann.get("images").keys())
num_images = len(img_files)

for img_idx in range(num_images):

    img_file = list(ann.get("images").keys())[img_idx]
    print(img_file)

    img = ann.get("images")[img_file]
    words = img["words"]


    def mouse_event(event, x, y, flags, param):
        global words, nw, nh, w, h

        if event == cv2.EVENT_LBUTTONDBLCLK:
            x = x/nh * h
            y = y/nw * w
            keys = words.keys()
            for k in keys:
                word = words[k]
                pts = np.array(word['points']).astype(np.int32)
                result = cv2.pointPolygonTest(pts, (x,y), False)
                if result > 0:
                    print(k, word['transcription'])
                    break

    # image load
    image = cv2.imread("./img/collage/"+img_file)
    canvas = np.zeros_like(image)

    # box color
    colors = {"default" : (0,0,255), "masked" : (255,0,0), "maintable" : (0,50,50), "stamp" : (0,255,0)}

    # draw boxes
    keys = words.keys()
    for k in keys:
        word = words[k]
        pts = np.array(word['points']).astype(np.int32)
        
        # select color
        tags = word['tags']
        color = colors['default']
        if "masked" in tags:
            color = colors["masked"]
        if "maintable" in tags:
            #continue
            color = colors["maintable"]
        if "stamp" in tags:
            color = colors["stamp"]
        mask_canvas = np.zeros_like(image)
        cv2.fillPoly(mask_canvas, [pts], color)
        cv2.putText(mask_canvas, k,(pts[0][0], pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,0), 2)
        canvas = cv2.addWeighted(canvas, 1, mask_canvas, 1, 0)

    image = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)

    # resize
    h = img['img_h']
    w = img['img_w']
    ratio = w / h
    nh = 1000
    nw = nh * ratio
    image = cv2.resize(image, (int(nw), nh))

    # show image
    cv2.imshow(img_file, image)
    cv2.setMouseCallback(img_file, mouse_event, image)
    cv2.waitKey(0)