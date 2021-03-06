def ParseAttributes(infname):
	f = open(infname,'r')
	attr,attrnames,tests,attrvals = {},[],[],{}
	attrnum = int(f.readline())
	for i in xrange(attrnum):
		fWords = f.readline().split()
		attrnames.append(fWords[0])
		attr[fWords[0]] = [i,fWords[1],fWords[2]]
		attrvals[fWords[1]],attrvals[fWords[2]] = 1,0
	num = attr[f.readline().strip()][0]
	testnum = int(f.readline())
	for i in xrange(testnum):
		fWords = f.readline().split()
		test = [0 for i in xrange(attrnum)]
		for j in xrange(attrnum):
			attrib = fWords[j][:fWords[j].find('=')]
			value = fWords[j][fWords[j].find('=')+1:len(fWords[j])]
			test[ attr[attrib][0] ] = attrvals[value]
		tests.append(test)
	f.close()
	return [attrnames,attr,tests,num]

def gain(tests,attrnum,num):
	import math
	def entropy(array):
		def log2(x): return math.log(x)/math.log(2)
		neg = float(len(filter(lambda x:(x[num]==0),array)))
		tot = float(len(array))
		if ((neg==tot) or (neg==0)): return 0
		return -(neg/tot)*log2(neg/tot)-((tot-neg)/tot)*log2((tot-neg)/tot)
	res = 0
	for i in xrange(2):
		arr = filter(lambda x:(x[attrnum]==i),tests)
		res += entropy(arr)*len(arr)/float(len(tests))
	return entropy(tests)-res

def ID3(tests,num,f,tabnum,usedattr,attrnames,attr):
	def findgains(x):
		if usedattr[x]: return 0
		return gain(tests,x,num)
	def fwriteline(x): f.write('\t'*tabnum+x+'\n')
	def majority():
		neg = len(filter(lambda x:(x[num]==0),tests))
		pos = len(filter(lambda x:(x[num]==1),tests))
		if (neg>pos): return '0'
		else: return '1'
	gains = map(findgains,xrange(len(tests[0])))
	maxgain = gains.index(max(gains))
	if (gains[maxgain]==0):
		fwriteline(majority())
		return
	arrpos=filter(lambda x:(x[maxgain]==1),tests)
	arrneg=filter(lambda x:(x[maxgain]==0),tests)
	newusedattr=[(usedattr[i] or (i==maxgain)) for i in xrange(len(usedattr))]
	fwriteline(attrnames[maxgain]+'='+attr[attrnames[maxgain]][1])
	if (len(arrpos)==0):
		fwriteline('\t'+majority())
	else:
		ID3(arrpos,num,f,tabnum+1,newusedattr,attrnames,attr)
	fwriteline(attrnames[maxgain]+'='+attr[attrnames[maxgain]][2])
	if (len(arrneg)==0):
		fwriteline('\t'+majority())
	else:
		ID3(arrneg,num,f,tabnum+1,newusedattr,attrnames,attr)	
	
def applyID3(infname,outfname):
	parsed = ParseAttributes(infname)
	attrnames,attr,tests,num = parsed[0],parsed[1],parsed[2],parsed[3]
	f = open(outfname,'w')
	usedattr = [(i==num) for i in xrange(len(attr))]
	ID3(tests,num,f,0,usedattr,attrnames,attr)
applyID3("in.txt","out.txt")
