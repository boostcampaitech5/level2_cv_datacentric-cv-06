#-*-coding:utf-8-*-
import json

train = open("2000.json", "rt", encoding='utf-8')
train = json.load(train)
anno = open("2010.json", "rt", encoding='utf-8')
anno = json.load(anno)

print(len(train["images"].keys()))
for img in anno["images"].keys():
    train["images"][img] = anno["images"][img]
    
print(len(train["images"].keys()))
new_train = open("train.json", "wt", encoding='utf-8')
json.dump(train, new_train, indent="\t", ensure_ascii=False)