from shapely.geometry import Polygon
import glob
import os
import json
from collections import defaultdict
#점수 높은 순
input1 = r'C:\Users\hyungu_lee\PycharmProjects\ocr\data\medical\ufo\output (9).csv' #기준
input2 = r'C:\Users\hyungu_lee\PycharmProjects\ocr\data\medical\ufo\output (10).csv'
input3 = r'C:\Users\hyungu_lee\PycharmProjects\ocr\data\medical\ufo\output (11).csv'
input_img = r'C:\Users\hyungu_lee\PycharmProjects\ocr\data\medical\img/test'
output = r'C:\Users\hyungu_lee\PycharmProjects\ocr\data\medical\ufo\0601FFensemble1_t_7.csv'

iou_thres=0.7
drop_single = True

def iou_cal(pg1,pg2):
    intersect = pg1.intersection(pg2).area
    union = pg1.union(pg2).area
    return intersect / union

with open(input1, encoding="utf-8") as file:
    ann1 = json.load(file)
with open(input1, encoding="utf-8") as file:
    annout = json.load(file)
with open(input2, encoding="utf-8") as file:
    ann2 = json.load(file)
with open(input3, encoding="utf-8") as file:
    ann3 = json.load(file)

#이미지 파일명 리스트
imgs = [os.path.basename(x) for x in glob.glob(input_img +'/*')]
#new json 생성
for i in imgs:
    while annout['images'][i]['words']:
        first_key = next(iter( annout['images'][i]['words']))
        del annout['images'][i]['words'][first_key]
new_output = annout

asdf=-1
for img in imgs:
    asdf +=1
    print(asdf)
    img_count = 0
    words1 = ann1["images"][img]["words"]
    words2 = ann2["images"][img]["words"]
    words3 = ann3["images"][img]["words"]

    while words1:
        #print(len(words1))
        count = 0
        first_key = next(iter(words1))
        word1 = words1[first_key]
        pt1 = word1['points']
        pg1 = Polygon(pt1)
        for word2 in words2:
            pt2 = words2[word2]['points']
            pg2 = Polygon(pt2)
            if iou_cal(pg1,pg2)>= iou_thres:
                count +=2
                new = words1[first_key]#['points']
                del ann2["images"][img]["words"][word2]
                del ann1["images"][img]["words"][first_key]
                break
        else:
            count +=1
            new = words1[first_key]#['points']
            del ann1["images"][img]["words"][first_key]

        for word3 in words3:
            pt3 = words3[word3]['points']
            pg3 = Polygon(pt3)
            if iou_cal(pg1, pg3) >= iou_thres:
                count += 1
                del ann3["images"][img]["words"][word3]
                break
        else:
            pass
        if drop_single:
            if count >= 2:
                new_output["images"][img]['words'][str(img_count)]=new
                img_count +=1
                #print(new_output)
        else:
            new_output["images"][img]['words'][str(img_count)]=new
            img_count += 1
            #print(new_output)

    while words2:
        #print(len(words2))
        count = 0
        first_key = next(iter(words2))
        word2 = words2[first_key]
        pt2 = word2['points']
        pg2 = Polygon(pt2)
        for word3 in words3:
            pt3 = words3[word3]['points']
            pg3 = Polygon(pt3)
            if iou_cal(pg2,pg3)>= iou_thres:
                count +=2
                new = words2[first_key]#['points']
                del ann3["images"][img]["words"][word3]
                del ann2["images"][img]["words"][first_key]
                break
        else:
            count +=1
            new = words2[first_key]#['points']
            del ann2["images"][img]["words"][first_key]
        if drop_single:
            if count >= 2:
                new_output["images"][img]['words'][str(img_count)]=new
                img_count +=1
                #print(new_output)
        else:
            new_output["images"][img]['words'][str(img_count)]=new
            img_count += 1
            #print(new_output)
    if drop_single:
        pass
    else:
        while words3:
            #print(len(words2))
            count = 0
            first_key = next(iter(words3))
            word3 = words3[first_key]
            new = words3[first_key]#['points']
            del ann3["images"][img]["words"][first_key]
            #print(new)
            new_output["images"][img]['words'][str(img_count)] = new
            img_count += 1
    #print(new_output)
with open(output, 'w', encoding="utf-8") as f:
    json.dump(new_output, f, indent=4)




    #
    # 2 3 비교
    # 같은 거 있으면 2기중 추가하고 지우기 ,count +=2
    # 없으면 추가하고 count += 1
    # drop_single 트루면 2만 저장, 아니면 다 저장
    #
    # 3에서
    # drop_single 트루면 다 저장
    # 아니면 다 폐기
    #
    #
    # 1 2 비교
    # 같은거 있으면 1기준 추가하고 지우기 count +=2
    #
    # 없으면 1추가하고 지우기 count +=1
    #
    #
    # 추가 랑 3 비교
    # 같은거 있으면 3 지우기 count += 1
    #
    # 없으면 패스 count += 0
    #
    # drop sing 트루면 count 23 만 저장
    # false 면 다 저장
        
#같은 이미지별 비교
'''
for img in
drop_single = True
csvs = [input1,input2,input3]
polygon1 = Polygon([(0, 0), (1, 1), (1, 0)])
polygon2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
intersect = polygon1.intersection(polygon2).area
union = polygon1.union(polygon2).area
iou = intersect / union
print(iou)  # iou = 0.5
'''