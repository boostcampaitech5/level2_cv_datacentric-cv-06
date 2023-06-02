import cv2
import json
import numpy as np

'''
img_idx 변경 or img_file을 원하는 이미지 파일명으로 변경하여 사용

masked : 파란색
maintable : 노란색
stamp : 초록색
그 외 : 빨간색
'''

# 경로 설정

with open("./data/medical/ufo\public6_cleaned50_title28_clean5.csv", encoding="utf-8") as file:
    ann = json.load(file)
print(len(ann['images']))
for i in ann['images']:
    print(i)
    print(len(ann['images'][i]['words']))
#'train' 'test'
mode = 'test'
img_idx = 0

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
while 1:
    img_file = list(ann.get("images").keys())[img_idx]
    img = ann.get("images")[img_file]
    words = img["words"]
    # image load
    image = cv2.imread("./data/medical/img/test/"+img_file)
    canvas = np.zeros_like(image)
    print(img_file)
    # box color
    colors = {"default" : (0,0,255), "masked" : (255,0,0), "maintable" : (0,50,50), "stamp" : (0,255,0)}

    # draw boxes
    keys = words.keys()
    for k in keys:
        word = words[k]
        pts = np.array(word['points']).astype(np.int32)

        # select color
        if mode == 'train':
            tags = word['tags']
            color = colors['default']
            if "masked" in tags:
                color = colors["masked"]
            if "maintable" in tags:
                #continue
                color = colors["maintable"]
            if "stamp" in tags:
                color = colors["stamp"]
        elif mode == 'test':
            color = colors["maintable"]
        mask_canvas = np.zeros_like(image)
        cv2.fillPoly(mask_canvas, [pts], color)
        canvas = cv2.addWeighted(canvas, 1, mask_canvas, 1, 0)

    image = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)

    # resize
    '''
    ratio = img['img_w'] / img['img_h']
    nh = 1000
    nw = nh * ratio
    image = cv2.resize(image, (int(nw), nh))
    '''
    h, w, _ = image.shape
    ratio = w/h
    nh = 1200
    nw = nh * ratio
    image = cv2.resize(image, (int(nw), nh))
    # show image
    cv2.imshow(img_file, image)
    cv2.setMouseCallback(img_file, mouse_event, image)
    cvkey = cv2.waitKeyEx(0)
    print(cvkey)
    if cvkey == 0x270000:  # 오른쪽 방향키
        img_idx +=1
    elif cvkey == 0x250000:  # 왼쪽 방향키
        img_idx -=1
    cv2.destroyAllWindows()