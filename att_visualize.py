name = '0540_4'

lines = [line.strip() for line in open("./save/att_vis/att_vis_%s.tsv"%name).readlines()]

def make_span(content, weights):
	line = ""
	words = content.split()
	weights = [float(i) for i in weights.split()]
	for i in range(len(words)):
		line+='<span style="background-color:rgba(0, 85, 160,{});padding:0 10px; float:left;">{}</span>'.format(1.5*weights[i], words[i])
	return line

divs = []
for line in lines:
	label, pred, content, weights = line.split("\t")
	line = '<div style="border:1px solid; padding:5px; max-width:400px; width:100%; overflow:auto; height:auto;"><span><b>Label: {}, Pred: {}</b></span><br><br>{}</div><br>'.format(label, pred, make_span(content, weights))
	# print line
	divs.append(line)

html = "<html style='width:100%'><body>{}</body></html>".format("".join(divs))
with open("visualize_%s.html"%name,"w") as f:
	f.write(html)
