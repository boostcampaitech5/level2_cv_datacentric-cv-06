import cv2
import json
import numpy as np

'''
img_idx 변경 or img_file을 원하는 이미지 파일명으로 변경하여 사용

masked : 파란색
maintable : 노란색
stamp : 초록색
그 외 : 빨간색

ctrl + 좌클릭으로 stamp tag 추가
'''

# 경로 설정
img_idx = 150
font_size = 0.5
with open("./data/medical/ufo/new_annotation.json", encoding="utf-8") as file:
    ann = json.load(file)

stamp_list = []


def mouse_event(event, x, y, flags, param):
    global words, nw, nh, w, h, ann, image, canvas, colors
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
    elif event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
        x = x/nh * h
        y = y/nw * w
        keys = words.keys()
        for k in keys:
            word = words[k]
            pts = np.array(word['points']).astype(np.int32)
            result = cv2.pointPolygonTest(pts, (x,y), False)
            if result > 0:
                stamp_list.append(k)
                word['tags'].append("stamp")
                ann['images'][img_file]['words'][k] = word
                print(k, word['transcription'], "-> add 'stamp' tag")
                cv2.fillPoly(canvas, [pts], colors['stamp'])
                image_mix = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)
                # resize
                h = img['img_h']
                w = img['img_w']
                ratio = w / h
                nh = 1000
                nw = nh * ratio
                image_resize = cv2.resize(image_mix, (int(nw), nh))

                # show image
                cv2.imshow(img_file, image_resize)
                cv2.setMouseCallback(img_file, mouse_event, image_resize)
                cv2.waitKey(0)
                break   


img_file = list(ann.get("images").keys())[img_idx]
img = ann.get("images")[img_file]
words = img["words"]

# image load
image = cv2.imread("./data/medical/img/train/"+img_file)
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
    cv2.fillPoly(canvas, [pts], color)
    #cv2.putText(canvas, k,(pts[0][0], pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,0), 2)

image_mix = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)
# resize
h = img['img_h']
w = img['img_w']
ratio = w / h
nh = 1000
nw = nh * ratio
image_resize = cv2.resize(image_mix, (int(nw), nh))

# show image
cv2.imshow(img_file, image_resize)
cv2.setMouseCallback(img_file, mouse_event, image_resize)
cv2.waitKey(0)



print("image close")
with open('./data/medical/ufo/new_annotation.json', 'w', encoding="utf-8") as file:
    json.dump(ann, file, indent=4, ensure_ascii=False)
print(img_file, stamp_list)

