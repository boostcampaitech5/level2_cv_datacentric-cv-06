import os
import json

"""
ufo.py으로 생성한 ufo 형식의 json 파일을 하나로 합칩니다.
"""

json_rt = "./jsons/2000"
ufo_rt = "./ufos/2000"
json_list = os.listdir(json_rt)

print(len(json_list))

for j in json_list:

    with open(os.path.join(json_rt, j), encoding='utf-8') as f:
        original = json.load(f)

    ufo = {"images": {}}

    ufo["images"][original["images"][0]["image.file.name"]] = {
        "paragraphs": {},
        "words": {}
    }
    words = {}
    for anno in original["annotations"]:
        id = str(anno["id"] + 1).zfill(4)
        anno["annotation.bbox"] = [float(i) for i in anno["annotation.bbox"]]
        points = [
            [anno["annotation.bbox"][0], anno["annotation.bbox"][1]],
            [anno["annotation.bbox"][0] + anno["annotation.bbox"][2], anno["annotation.bbox"][1]],
            [anno["annotation.bbox"][0] + anno["annotation.bbox"][2], anno["annotation.bbox"][1] + anno["annotation.bbox"][3]],
            [anno["annotation.bbox"][0], anno["annotation.bbox"][1] + anno["annotation.bbox"][3]]
        ]
        words.update({
            id: {
                "transcription": anno["annotation.text"],
                "points": points,
                "orientation": "Horizontal",
                "language": None,
                "tags": [],
                "confidence": None,
                "illegibility": False
            }})
    ufo["images"][original["images"][0]["image.file.name"]]["words"] = words

    with open(os.path.join(ufo_rt, j), 'w', encoding='utf-8') as f:
        json.dump(ufo, f, ensure_ascii=False, indent="\t")