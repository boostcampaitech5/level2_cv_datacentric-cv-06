import json

"""
new_viewer.py와 같이 실행하기
즉시 stamp 태그를 추가할 수 있습니다.
"""

with open("./annotation.json", encoding="utf-8") as file:
    ann = json.load(file)

img_files = list(ann.get("images").keys())

ufo = {"images": {}}

for img_idx in range(51):

    img_file = list(ann.get("images").keys())[img_idx]
    print(img_file)

    img = ann.get("images")[img_file]
    words = img["words"]
    keys = words.keys()

    order = ""
    while order != "q":
        order = input("number to tag stamp: ")
        if order == "q":
            break
        order = order.zfill(4)
        
        if order in keys:
            words[order]["tags"].append("stamp")
            print("stamped")
    ufo["images"].update({img_file: img})

with open("./new_annotation.json", "wt", encoding="utf-8") as file:
    json.dump(ufo, file, indent="\t", ensure_ascii=False)