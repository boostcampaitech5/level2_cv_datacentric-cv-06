import json

json_path = r'C:\Users\hyungu_lee\PycharmProjects\ocr\data\medical\ufo\new_train_cleaned_2810_3213.json'
output_path = r'C:\Users\hyungu_lee\PycharmProjects\ocr\data\medical\ufo\new_train_cleaned_2810_3213_done.json'

del_img = ['drp.en_ko.in_house.deepnatural_002811.jpg',
           'drp.en_ko.in_house.deepnatural_002814.jpg',
           'drp.en_ko.in_house.deepnatural_002854.jpg',
           'drp.en_ko.in_house.deepnatural_002896.jpg',
           'drp.en_ko.in_house.deepnatural_002921.jpg',
           'drp.en_ko.in_house.deepnatural_002922.jpg',
           'drp.en_ko.in_house.deepnatural_002923.jpg',
           'drp.en_ko.in_house.deepnatural_002948.jpg',
           'drp.en_ko.in_house.deepnatural_002991.jpg',
           'drp.en_ko.in_house.deepnatural_003004.jpg',
           'drp.en_ko.in_house.deepnatural_003008.jpg',
           'drp.en_ko.in_house.deepnatural_003044.jpg',
           'drp.en_ko.in_house.deepnatural_003071.jpg',
           'drp.en_ko.in_house.deepnatural_003140.jpg',
           'drp.en_ko.in_house.deepnatural_003142.jpg',
           'drp.en_ko.in_house.deepnatural_003143.jpg',
           'drp.en_ko.in_house.deepnatural_003151.jpg',
           'drp.en_ko.in_house.deepnatural_003174.jpg',
           'drp.en_ko.in_house.deepnatural_003182.jpg',
           'drp.en_ko.in_house.deepnatural_003191.jpg'
           ]

to_stamp = {'drp.en_ko.in_house.deepnatural_002810.jpg':['0562','0563','0276','0561','0560','0280','0284','0286'],
            'drp.en_ko.in_house.deepnatural_002834.jpg':['0114','0118','0119','0123','0120','0127','0128','0129','0130','0134','0138','0139','0140','0141','0416','0417','0418'],
            'drp.en_ko.in_house.deepnatural_002926.jpg':['0033'],
            'drp.en_ko.in_house.deepnatural_003120.jpg':['0295','0297','0296','0299','0298'],
            'drp.en_ko.in_house.deepnatural_003175.jpg':['0269','0268'],
            'drp.en_ko.in_house.deepnatural_003094.jpg':['0154','0157']
            }

with open(json_path, encoding="utf-8") as f:
    ann = json.load(f)
#del img
for i in del_img:
    del ann['images'][i]
#add stamp
for i in to_stamp:
    for j in to_stamp[i]:
        ann['images'][i]['words'][j]['tags'].append("stamp")

with open(output_path, 'w', encoding="utf-8") as f:
    json.dump(ann,f, indent=4)
