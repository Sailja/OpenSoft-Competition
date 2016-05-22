import  csv
import pickle

def getColHeadings(temp):
	tableH = []
	tableH.append('x')
	for color in temp[3]:
		print color
		tableH.append(color)
	print tableH
	tableH = ','.join(tableH)
	tableH += '\n'
	return tableH

fnamelist=[]
def listCSV(temp,i):
	file_name = "plot_"+str(i) + ".csv"
	
	csvfile = open("tablefolder/"+file_name,'w')
	fnamelist.append("tablefolder/"+file_name)
	csvfile.write(temp[0] + "\n")
	row1 = temp[1] + " vs " + temp[2] + "\n"
	csvfile.write(row1)

	numColors = len(temp[3])
	tableH = getColHeadings(temp)
	csvfile.write(tableH)
	print tableH
	print len(temp[4][0])

	i=0
	j=0 
	k=0
	while i<len(temp[4][0]):
		row = str(temp[4][0][i][0])
		j=0
		while j<numColors:
			row = str(row) + "," + str(temp[4][j][i][1])
			# print row
			j+=1
		print row
		row+='\n'
		csvfile.write(row)
		i+=1


# temp = ['title', 'xname', 'yname', ['red', 'blue'], [[('x1','y1'),('x2','y2')],[('x1','2'),('x2','4')]]]
# temp2 = ['title2', 'xname2', 'yname2', ['red', 'blue'], [[('x1','y1'),('x2','y2')],[('x1','2'),('x2','4')]]]
# listCSV(temp)
Values = pickle.load( open( "table.p", "rb" ) )
# Values = [[]]
# temp=
# Values.append(temp)
# Values.append(temp2)

print Values
# Values.pop(0)
i=1
for temp in Values:
	listCSV(temp,i)
	i+=1

pickle.dump(fnamelist, open('filelist.p', 'wb'))