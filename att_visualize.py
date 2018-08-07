from collections import Counter

name = '0111'

lines = [line.strip() for line in open("./save/att_vis/att_vis_%s.tsv"%name).readlines()]

def make_span(words, weights, color, counter):
	line = ""
	weights = [float(i) for i in weights]
	rgb = [0, 85, 160]
	rgb += rgb
	for i in range(len(words)):
		line+='<span style="background-color:rgba({}, {}, {}, {});padding:0 10px; float:left;">{} ({})</span>'.format(rgb[color], rgb[color+1], rgb[color+2], 0.7*weights[i], words[i], counter[words[i]])
	return line

divs = []
for line in lines:
	label, pred, content, w1, w2, w3 = line.split("\t")
	words = content.split()
	w1 = w1.split()
	w2 = w2.split()
	w3 = w3.split()

	brand_counter = Counter()
	brand_counter += Counter(words)
	
	content_set = []
	weight1, weight2, weight3 = [],[],[]
	for i, c in enumerate(words):
		if not c in content_set:
			content_set.append(c)
			weight1.append(w1[i])
			weight2.append(w2[i])
			weight3.append(w3[i])
	
	label = label.replace('[','').replace(']','').replace(",",'').replace("'",'').split()
	pred = pred.replace('[','').replace(']','').replace(",",'').replace("'",'').split()
	line = '<div style="border:1px solid; padding:5px; max-width:400px; width:30%; overflow:auto; height:auto; float:left;"><span><b>Label: {}, Pred: {}</b></span><br><br>{}</div><div style="border:1px solid; padding:5px; max-width:400px; width:30%; overflow:auto; height:auto; float:left;"><span><b>Label: {}, Pred: {}</b></span><br><br>{}</div><div style="border:1px solid; padding:5px; max-width:400px; width:30%; overflow:auto; height:auto; float:left;"><span><b>Label: {}, Pred: {}</b></span><br><br>{}</div><div style="clear:both;"></div><br>'.format(label[0], pred[0], make_span(content_set, weight1, 0, brand_counter), label[1], pred[1], make_span(content_set, weight2, 1, brand_counter), label[2], pred[2], make_span(content_set, weight3, 2, brand_counter))
	# print line
	divs.append(line)

html = "<html style='width:100%'><body>{}</body></html>".format("".join(divs))
with open("visualize_%s.html"%name,"w") as f:
	f.write(html)


