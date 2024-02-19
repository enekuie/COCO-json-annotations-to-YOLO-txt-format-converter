import json
import glob
from pathlib import Path
import pandas as pd

input_path = "./coco/labels/train/*.json"
output_path = "./yolo/labels/train/"

files = glob.glob(input_path)
for file in files:
    name = Path(file).stem
    with open(file, "r") as f:
        data = json.load(f)
        cls = []
        bbox = []
        category_to_id = {'two_side': 0, 'four_side': 1, 'without_side': 2} #change it according your dataset
        h = data["imageHeight"]
        w = data["imageWidth"]
        for items in data["shapes"]:
            cls.append(items['label'])
            classes = [category_to_id.get(e, e) for e in cls]
            bbox.append(items["points"])
        xxy = []
        for box in bbox:
            x1 = box[0][0]
            x2 = box[1][0]
            y1 = box[0][1]
            y2 = box[1][1]
            xc = round(((x1+x2)/2)/w, 5)
            yc = round(((y1+y2)/2)/h, 5)
            w1 = round(abs((x1-x2))/w, 5)
            h1 = round(abs((y1-y2))/h, 5)
            xy = [xc, yc, w1, h1]
            xxy.append(xy)
        
        with open(output_path + name + '.txt', 'w') as t:
            df = pd.DataFrame(list(zip(classes, xxy)))
            df[1] = df[1].astype(str).str[1:-1]
            df[1] = df[1].str.replace(',', '')
            dfa = df.copy(deep=True)
            items = []
            items_to_change = []
            d = []
            d1 = []
            cc1 = []
            for i in dfa.loc[:, 1]:
                if len(i) < 31:
                    items.append(i)
                    c = i.split(' ')
                    k = c.copy()
                    for cc in c:
                        if len(cc) < 7:
                            n = 7-len(cc)
                            c1 = cc + "0"*n
                            cc1.append(c1)
                            for j in cc1:
                                if not cc == j:
                                    ind = c.index(cc)
                                    k[ind] = j
                                    dd = " ".join(k)
                                else:
                                    continue
                                d1.append(dd)
                    items_to_change.append(dd)

                dfb = dfa.loc[:, :].replace(items, items_to_change)
                
            dfAsString = dfb.to_string(header=False, index=False)
            # t.write(dfAsString)
            tt = [list(x) for x in df.to_records(index=False)]
            for i in tt:
                classs = i[0]
                p = i[1]
                xc, yc, ww, hh = tuple([float(x) for x in p.split(' ')])
                t.write("%s %.6f %.6f %.6f %.6f\n" % (classs, xc, yc, ww, hh))
        t.close()
    f.close()
