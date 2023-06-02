import os
import json

"""
공공기관 데이터를 합치다 보면 실수로 사진이 누락되는 경우가 있습니다.
데이터가 충분히 많기 때문에 놓치면 그냥 그 사진을 빼고 합치는 것이 좋습니다.
이 코드는 누락된 사진을 찾아서 json 파일에서 삭제하는 코드입니다.
"""

data_list = os.listdir("./data/2000")
print(len(data_list))

with open("final.json", encoding='utf-8') as f:
    final = json.load(f)

print(len(list(final["images"].keys())))
images = list(final["images"].keys())

for img in images:
    if img not in data_list:
        del final["images"][img]

print(len(list(final["images"].keys())))

with open("final.json", 'w', encoding='utf-8') as f:
    json.dump(final, f, ensure_ascii=False, indent="\t")