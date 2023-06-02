import json

"""
new_viewer.py에서 이상했던 사진들을 삭제할 수 있습니다.
"""

with open("./new_annotation.json", encoding="utf-8") as file:
    ann = json.load(file)

img_files = list(ann.get("images").keys())

order = ""
while order != "q":
    order = input("Image to delete: ")
    if order == "q":
        break
    order = "drp.en_ko.in_house.deepnatural_00" + order + ".jpg"
    if order in img_files:
        del ann["images"][order]
        print("deleted")


with open("./new_annotation.json", "wt", encoding="utf-8") as file:
    json.dump(ann, file, indent="\t", ensure_ascii=False)