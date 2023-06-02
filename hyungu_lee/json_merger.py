import json
import os
json_dir = r'C:\Users\hyungu_lee\PycharmProjects\ocr\data\medical\ufo/'
new_path = r'C:\Users\hyungu_lee\PycharmProjects\ocr\data\medical\ufo/title_stamp.json'

basename = ['title1.json','title2.json','title3.json','title4.json','title5.json','title6.json','stamp1.json','stamp2.json']
asdf= dict()
asdf ={'images':dict()}
for i in basename:
    path = json_dir+i
    with open(path,encoding='utf-8') as f:
        ann = json.load(f)
    asdf['images'].update(ann['images'])

with open(new_path, 'w',  encoding="utf-8") as f:
    json.dump(asdf,f, indent=4, ensure_ascii=False)