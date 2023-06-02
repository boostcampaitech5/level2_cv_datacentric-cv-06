import json

"""
ai_hub 공공기관 데이터에서 사용하는 annotation 형식을 ufo 형식으로 변환합니다.
각 이미지마다 하나의 json 파일을 생성합니다. 후에 ufos.py에서 하나로 합칩니다.
"""

with open('jsons/5350034-2011-0001-0018.json', encoding='utf-8') as f:
    original = json.load(f)

ufo = {"images": {}}

ufo["images"][original["images"][0]["image.file.name"]] = {
    "paragraphs": {},
    "words": {}
}
words = {}
for anno in original["annotations"]:
    id = str(anno["id"]).zfill(4)
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

with open('ufos/5350034-2011-0001-0018.json', 'w', encoding='utf-8') as f:
    json.dump(ufo, f, ensure_ascii=False, indent="\t")