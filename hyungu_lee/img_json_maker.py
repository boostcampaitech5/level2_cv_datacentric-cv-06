import cv2
import json
import numpy as np
import os
'''
img_idx 변경 or img_file을 원하는 이미지 파일명으로 변경하여 사용

masked : 파란색
maintable : 노란색
stamp : 초록색
그 외 : 빨간색
'''

# 경로 설정
new_img_path = r'C:\Users\hyungu_lee\PycharmProjects\ocr\data\medical\img/stamp2.jpg'
new_json_path = r'C:\Users\hyungu_lee\PycharmProjects\ocr\data\medical\ufo/stamp2.json'
with open("./data/medical/ufo/train.json", encoding="utf-8") as file:
    ann = json.load(file)
#'train' 'test'
mode = 'train'
img_idx = 69


def onmouse(event, x, y, flags, param):
    global isDragging, x0, y0, image,row_image, img_file, words, offsetx, offsety, new_img, w, h, nw,nh, new_key
    if event == cv2.EVENT_LBUTTONDOWN:
        isDragging = True
        x0 = x
        y0 = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            img_draw = image.copy()
            cv2.rectangle(img_draw, (x0, y0), (x, y), blue, 2)
            cv2.imshow(img_file, img_draw)
    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            w_ = x - x0
            h_ = y - y0
            if w_ > 0 and h_ > 0:
                img_draw = image.copy()
                cv2.rectangle(img_draw, (x0, y0), (x, y), red, 2)
                roi = row_image[y0:y0 + h_, x0:x0 + w_]
                if (w_ >= 1024) or (h_ >= 1200):
                    print('too big')
                    pass
                else:
                    if (offsetx + w_ <= 1024) and (offsety + h_ <= 1200):
                        # xy 둘 다 offset 초과 안함.
                        # 이미지 그리기
                        new_img[offsety:offsety + h_,offsetx:offsetx + w_ ] = roi
                        keys = words.keys()
                        for k in keys:
                            word = words[k]
                            pts = np.array(word['points']).astype(np.int32)
                            ptx = pts[:,0]*(nw/w)
                            pty = pts[:,1]*(nh/h)
                            ptx_b = (x0 <= ptx).all() and (ptx <= x0+w_).all()
                            pty_b = (y0 <= pty).all() and (pty <= y0+h_).all()
                            if pty_b and ptx_b:
                                nptx = (ptx - x0) + offsetx
                                npty = (pty - y0) + offsety
                                npt = np.vstack((nptx, npty)).transpose()
                                new_json['images'][new_img_base]['words']['{0:04d}'.format(new_key)]=word
                                new_json['images'][new_img_base]['words']['{0:04d}'.format(new_key)]['points']=npt.tolist()
                                new_key +=1
                                print(new_json)
                        offsety += h_
                    else:
                        offsety = 0
                        cv2.imshow('newimg',new_img)
                        cv2.imwrite(new_img_path,new_img)
                        with open(new_json_path, 'w', encoding='utf-8') as f:
                            json.dump(new_json,f,indent=4, ensure_ascii=False)
            else:
                cv2.imshow(img_file, image)
                print('drag should start from left-top side')

#set global var
isDragging = False
x0, y0, w, h = -1, -1, -1, -1
blue, red = (255, 0, 0), (0, 0, 255)
new_img = np.full((1200,1024,3),170, np.uint8)
offsetx, offsety = 500, 0
new_key = 1
#init json
new_img_base = str(os.path.basename(new_img_path))
new_json = {'images':{new_img_base:{"paragraphs": {},"words": {}}}}
print(new_json)
while 1:
    print(img_idx)
    img_file = list(ann.get("images").keys())[img_idx]
    img = ann.get("images")[img_file]
    words = img["words"]
    # image load
    image = cv2.imread("./data/medical/img/train/"+img_file)
    row_image = image
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
        canvas = cv2.addWeighted(canvas, 1, mask_canvas, 1, 0)

    image = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)

    # resize

    h, w, _ = image.shape
    ratio = w/h
    nh = 1200
    nw = nh * ratio
    image = cv2.resize(image, (int(nw), nh))
    row_image = cv2.resize(row_image, (int(nw), nh))
    # show image
    cv2.imshow(img_file, image)
    cv2.setMouseCallback(img_file, onmouse, image)
    cvkey = cv2.waitKeyEx(0)

    if cvkey == 0x270000:  # 오른쪽 방향키
        img_idx +=1
    elif cvkey == 0x250000:  # 왼쪽 방향키
        img_idx -=1
    cv2.destroyAllWindows()