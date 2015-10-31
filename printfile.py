import csv

def writecsv(name, curves):
    with open(name, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["answers","run","fp","fn","tp","tn","fpc","fnc","tpc","tnc","F2","sensitivity","specificity","precision","accuracy","balanced_accuracy","utility19"])
        for i in xrange(0,len(curves)):
            for r in curves[i]:
                writer.writerow([r[0],i,r[1]['fp'],r[1]['fn'],r[1]['tp'],r[1]['tn'],,r[1]['fpc'],r[1]['fnc'],r[1]['tpc'],r[1]['tnc'],r[1]['F2'],r[1]['sensitivity'],r[1]['specificity'],r[1]['precision'],r[1]['accuracy'],r[1]['balanced_accuracy'],r[1]['utility19']])
