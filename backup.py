import pytesseract
import requests
from PIL import Image
from PIL import ImageFilter
import cv2
import numpy as np
import re
import os
import sys
import string
import math
from collections import namedtuple
import operator
import pickle
# import division

#function to findthe maximum frequency pixels
def find_max(colors,n):
	count = 0
	max_color = []
	maxs = []
	freq = []
	
	i=0

	while i<int(n):
		#print i
		count = np.amax(colors)
		b,g,r = np.unravel_index(colors.argmax(), colors.shape)
		#print count 
		max_color.append(b)
		max_color.append(g)
		max_color.append(r)
		maxs.append(max_color)
		freq.append(count)
		#print str(b)+" "+str(g)+" "+str(r)
		colors[b][g][r] = 0
		count = -1
		max_color = []
		i+=1
	return (maxs, freq)

def getValues(t):
	val=[]
	for i in t.split():
		p=0
		if i[0]=='.' and len(i)==1:
			continue
		if i[0]=='-' and len(i)==1:
			continue
		if not(i[0].isdigit() or i[0]=='.' or i[0]=='-'):
			continue
		for j in i[1:]:
			if not (j.isdigit()):
				p=1
		if i[0]=='0' and len(i)>1:
			if i[1]!='.':
				i = i[:0]+'.'+i[1:]
		if p==0:
			val.append(i)

	print val
	return val

def process_image(img):
	image = Image.fromarray(img)
	#bg = Image.new("RGB", image.size, (255,255,255))
	#bg.paste(image,image)
	#image=bg
    #image.convert('RGB')
	image.filter(ImageFilter.SHARPEN)
	return pytesseract.image_to_string(image)

def otransform(l,x,y):
	if l:
		return Line(l.x1+x,y-l.y2,l.x2+x,y-l.y1)
	else:
		return Line(x,y,x,y)

# def getCand(newcand,thr):
# 	cand=[]
# 	for a,b in newcand:
# 		for c,d in cand:
# 			if a.x1>c.x1 and a.x2<c.x2 and abs(a.y1-c.y1)<thr:
# 				if b.y1>d.y1 and b.y2<d.y2 and abs(b.x1-d.x1)<thr:
# 					cand.append(newcand)


def bhar(im,type):
	bashCommand = "tesseract "+im+" image batch.nochop makebox"
	os.system(bashCommand)
	f = open("image.box", "r+").read().splitlines()
	rx = []
	for line in f:
		if line[0].isdigit():
			coord = [int(a) for a in line[1:].split()] 
			if type==1:
				rx.append(int((coord[0] + coord[2])/2))
			else:
				rx.append(int((coord[1] + coord[3])/2))

	x1=y1=x2=y2=0
	if type==1:
		if rx:
			x1 = min(rx)
			y1 = (coord[1] + coord[3])/2
			y2 = (coord[1] + coord[3])/2
			x2 = max(rx)
			return Line(min(x1,x2),y1,max(x1,x2),y2)
	else:
		if rx:
			x1 = (coord[0] + coord[2])/2
			y1 = min(rx)
			y2 = max(rx)
			x2 = (coord[0] + coord[2])/2
			return Line(x1,min(y1,y2),x2,max(y1,y2))


image = cv2.imread(sys.argv[1])
print image.shape
if min(image.shape[0],image.shape[1])> 10000:
	img = cv2.imread(sys.argv[1])
	z=1
else:
	z = math.ceil(10000/ (1.0*min(image.shape[0],image.shape[1])))
	print z
	bashCommand = "convert -resize "+ str(100*z) + "% " +sys.argv[1]+" zoom.jpg"
	os.system(bashCommand)
	img = cv2.imread("zoom.jpg")


print "Done"
					

testimg = np.zeros(img.shape, img.dtype)
testimg[:,:] = (255,255,255)
kernel = np.ones((5,5),np.float32)/25
img = cv2.filter2D(img,-1,kernel) 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
lowThresh = 0.5*high_thresh
edges = cv2.Canny(gray, lowThresh, high_thresh)
kernel = np.ones((3,3),np.uint8)
dilation = cv2.dilate(edges,kernel,iterations = 1)
im = cv2.bilateralFilter(dilation, 5, 17,17)
lines = cv2.HoughLinesP(im,1,np.pi/180,20,10)
hor = []
ver = []
thr=0.1*img.shape[0]
Line = namedtuple("Line", "x1 y1 x2 y2")
finallist=[]
print("Test 1 ")
h=0
v=0
for x1,y1,x2,y2 in lines[0]:
	if abs(y1-y2)<=5 and abs(x1-x2)>thr:
		hor.append(Line(min(x1,x2),y1,max(x1,x2),y2))
		cv2.line(testimg,(min(x1,x2),y1),(max(x1,x2),y2),(0,0,0),10)
		h+=1
	if abs(x1-x2)<=5 and abs(y1-y2)>thr:
		ver.append(Line(x1,min(y1,y2),x2,max(y1,y2)))
		cv2.line(testimg,(min(x1,x2),y1),(max(x1,x2),y2),(0,0,0),10)
		v+=1


# cv2.namedWindow('rotated', cv2.WINDOW_NORMAL)
# cv2.imshow("rotated", img)
# cv2.waitKey(0)
# cv2.namedWindow('houghlines3', cv2.WINDOW_NORMAL)
# cv2.imshow("houghlines3", testimg)
# cv2.waitKey(0)
print h,v 	
print("Test 2")
cand=[]
c=0
for a in hor:
	for b in ver:
		xdist = min(abs(a.x1-b.x1),abs(a.x2-b.x1))
		ydist = min(abs(b.y1-a.y1), abs(b.y2-a.y1))
		mx=(a.x1+a.x2)/2
		my=(b.y1+b.y2)/2
		lh= abs(a.x1-a.x2)
		ly= abs(b.y1-b.y2)
		if xdist <= 2 and ydist<=2 :
			if mx < b.x1:
				continue
			if my > a.y1:
				continue
			if abs(lh-ly)>2*min(lh,ly):
				continue
			c+=1
			cand.append((a,b))
			cv2.line(testimg,(a.x1,a.y1),(a.x2,a.y2),(0,0,0),10)
			cv2.line(testimg,(b.x1,b.y1),(b.x2,b.y2),(0,0,0),10)

print("Test 3")
print c
acc=0
newcand=[]
for a,b in cand:
	duplicate=0
	for c,d in newcand:
		if abs(c.y1-a.y1)<=20 and abs(c.x1-a.x1)<=20 and abs(c.x2-a.x2)<=20:
			if abs(d.x1-b.x1)<=20 and abs(d.y1-b.y1)<=20 and abs(d.y2-b.y2)<=20:
				duplicate=1


	if duplicate==1:
		continue
	acc+=1
	newcand.append((a,b))

print("Test 4")
print acc
# newcand = getCand(newcand)
hstrip = 150
vstrip = 200

plotnum=-1
plot_list=[]
for a,b in newcand:
	
	prevt1=None
	prevt3=None
	i=1
	while y1+i*hstrip<img.shape[0]:
		crop_img1 = img[a.y1:a.y1+i*hstrip, a.x1:a.x2]
		t1 = process_image(crop_img1)
		# cv2.namedWindow('rotated', cv2.WINDOW_NORMAL)
		# cv2.imshow("rotated", crop_img1)
		# cv2.waitKey(0)

		if t1:
			val=getValues(t1)
			if val:
				hstrip=i*hstrip
				break

		i+=1
		if i>=10:
			break
		print i
	i=1
	while b.x1-i*vstrip>0:
		crop_img3 = img[b.y1:b.y2, b.x1-i*vstrip:b.x1]
		t3 = process_image(crop_img3)
		if t3:
			val=getValues(t3)
			if val:
				vstrip=i*vstrip
				break

		i+=1
		if i>=10:
			break
		print i

	if not t1:
		if not t3:
			continue
	
	# print t1,t3
	plotnum+=1
	cv2.imwrite('hor.jpg',crop_img1)
	cv2.imwrite('ver.jpg',crop_img3)
	hline = bhar('hor.jpg',1)
	vline = bhar('ver.jpg',2) 
	hline = otransform(hline,a.x1,a.y1+hstrip)
	vline = otransform(vline,b.x1-vstrip,b.y2)
	xlabs = getValues(t1)
	ylabs = getValues(t3)
	# print hline, xlabs
	# print vline, ylabs
	y1=float(ylabs[0])
	y2=float(ylabs[-1])
	y0=vline.y2 + y2 * ((vline.y2-vline.y1)/(1.0*(y1-y2)))

	x1=float(xlabs[-1])
	x2=float(xlabs[0])
	x0= hline.x1 - x2 * ((hline.x2-hline.x1)/(1.0*(x1-x2)))

	dy = (vline.y2-vline.y1)/(y1-y2)
	# print vline.y2, vline.y1, y1,y2	
	dx = (hline.x2-hline.x1)/(x1-x2)

	prev=-1
	mingap=x1-x2
	for i in xlabs:
		if prev!=-1:
			newgap=float(i)-prev
			prev=float(i)
			mingap=min(mingap,newgap)
		else:
			prev=float(i)
	minpixgap= (mingap * dx)/10

	h=150
	v=250
	xstrip = img[a.y1+hstrip:a.y2+hstrip+h, a.x1:a.x2]
	ystrip = img[b.y1:b.y2, b.x1-200-v:b.x1-200]
	# cv2.namedWindow('rotated', cv2.WINDOW_NORMAL)
	# cv2.imshow("rotated", ystrip)
	# cv2.waitKey(0)

	e=max(ystrip.shape[0],ystrip.shape[1])
	limg = np.zeros((e,e,ystrip.shape[2]),ystrip.dtype)
	limg[:,:]=(255,255,255)
	limg[0:ystrip.shape[0],0:ystrip.shape[1]]=ystrip
	rows,cols = limg.shape[:2]
	M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
	ystrip = cv2.warpAffine(limg,M,(cols,rows))
	xname = process_image(xstrip)
	yname = process_image(ystrip)


	titlestrip = img[a.y1+hstrip+1.5*h:a.y2+hstrip+2.5*h, a.x1-200:a.x2+200]
	title = process_image(titlestrip)

	print xname
	print yname
	print title
	zoom=img
	image = cv2.imread(sys.argv[1])
	x_c1=int(a.x1/z)
	x_c2=int(a.x2/z)
	y_c1=int(b.y1/z)
	y_c2=int(b.y2/z)#2183#4270//4
	y_o= int(y0/z-y_c1)#2150-y_c1#4-y_c1
	dy/=1.0*z#5.35#30*1.0/4
	dx/=1.0*z#40#170*1.0/4
	x_o=int(x0/z-x_c1) #-2021-x_c1#-18991//4-x_c1
	gap=int(minpixgap/4)#2*170//40
	if gap==0:
		gap=10
	print dx,dy,x_o,gap
	crop_img = image[y_c1:y_c2, x_c1:x_c2]

	rows, cols, channels = crop_img.shape
	row = 0
	col = 0
	colors = np.zeros((256,256,256))
	for row in xrange(0,rows):
		for col in xrange(0,cols):
			pixel = crop_img[row,col]
			b = int(pixel[0])
			g = int(pixel[1])
			r = int(pixel[2])
			if not((b>200 and g>200 and r>200) or (b<50 and g<50 and r<50)or (abs(r-g)<30 and abs(g-b)<30 and abs(r-b)<30)):		
				colors[b][g][r]+=1;

	maxs, freq = find_max(colors, 100)

	boundaries=[maxs[0]]
	for (bi,gi,ri) in maxs:
		count=0
		for (bb,gb,rb) in boundaries:
			if not (abs(bb-bi)<50 and abs(gb-gi)<50 and abs(rb-ri)<50):
				count+=1
		if(count==len(boundaries)):
				boundaries.append((bi,gi,ri))
	num_plots=len(boundaries)
	plot_list.append([])
	
	li_x=[]
	li_y=[]
	noise_l=int(0.04*(crop_img.shape[0]))
	start_x_window =noise_l
	end_x_window = crop_img.shape[1]-noise_l
	start_y_window=noise_l
	end_y_window=crop_img.shape[0]-noise_l
	mask_b=cv2.inRange(crop_img,(0,0,0),(40,40,40))
	for i in range(start_x_window,end_x_window):
		for j in range(start_y_window,end_y_window):
			if (mask_b[j][i]==255):
				li_x.append(i)
				li_y.append(j)

	print min(li_x),max(li_x)
	dic={}
	origx = min(li_x)
	origy = min(li_y)
	lenx = max(li_x)
	leny = max(li_y)
	legendold = crop_img[min(li_y):max(li_y),min(li_x):max(li_x)]
	legend = mask_b[min(li_y):max(li_y),min(li_x)+10:max(li_x)-10]
	legend1 = crop_img[min(li_y):max(li_y),min(li_x)+10:max(li_x)-10]
	# cv2.namedWindow('rotated', cv2.WINDOW_NORMAL)
	# cv2.imshow("rotated", legendold)
	# cv2.waitKey(0)
	li_x2=[]
	li_y2=[]
	for i in range(0,legend.shape[1]):
		for j in range(0,legend.shape[0]):
			if (legend[j][i]==255):
				li_x2.append(i)
				li_y2.append(j)
	if min(li_x2)>10 or (lenx-leny-max(li_x2))>10:
		flag=1
	else:
		flag=0
	if flag==1:
		legend = legend1[min(li_y2):max(li_y2),min(li_x2):max(li_x2)]
		legendlines1=crop_img[origy:leny,origx:origx+min(li_x2)]
		legendlines2=crop_img[origy:leny,origx+max(li_x2):lenx]
		if legendlines1.shape[1]>legendlines2.shape[1]:
			legendlines = legendlines1
		else:
			legendlines = legendlines2
		dic={}
		for (b,g,r)in boundaries:
			lower = np.array((0 if (b-40<0) else b-40,0 if (g-40<0) else g-40,0 if (r-40<0) else r-40), dtype = "uint8")
			upper = np.array((255 if (b+40>255) else b+40,255 if (g+40>255) else g+40,255 if (r+40>255) else r+40), dtype = "uint8")
			for i in range(legendlines.shape[0]):
				for j in range(legendlines.shape[1]):
					if (legendlines[i][j]>lower).all() and (legendlines[i][j]<upper).all():
						dic[(b,g,r)]=j
		ex1=min(li_x2)+origx
		ey1=min(li_y2)+origy
		ex2=max(li_x2)+origx
		ey2=max(li_y2)+origy
	else:
		ex1=origx
		ey1=origy
		ex2=lenx
		ey2=leny
		legend = legendold
		dic={}
		legendlines1=crop_img[origy:leny,lenx:min(lenx+150,img.shape[1])]
		legendlines2=crop_img[origy:leny,max(origx-150,0):origx]
		for (b,g,r)in boundaries:
			lower = np.array((0 if (b-40<0) else b-40,0 if (g-40<0) else g-40,0 if (r-40<0) else r-40), dtype = "uint8")
			upper = np.array((255 if (b+40>255) else b+40,255 if (g+40>255) else g+40,255 if (r+40>255) else r+40), dtype = "uint8")
			for i in range(legendlines1.shape[0]):
				for j in range(legendlines1.shape[1]):
					if (legendlines1[i][j]>lower).all() and (legendlines1[i][j]<upper).all():
						dic[(b,g,r)]=j
		dic={}
		for (b,g,r)in boundaries:
			lower = np.array((0 if (b-40<0) else b-40,0 if (g-40<0) else g-40,0 if (r-40<0) else r-40), dtype = "uint8")
			upper = np.array((255 if (b+40>255) else b+40,255 if (g+40>255) else g+40,255 if (r+40>255) else r+40), dtype = "uint8")
			for i in range(legendlines1.shape[0]):
				for j in range(legendlines1.shape[1]):
					if (legendlines1[i][j]>lower).all() and (legendlines1[i][j]<upper).all():
						dic[(b,g,r)]=j		
			

			
	
	# cv2.imshow("rotated", legendlines1)
	# cv2.waitKey(0)
	# cv2.imshow("rotated", legendlines2)
	# cv2.waitKey(0)
	
	
	# print dic
	# print "DIC"
	sorted_x = sorted(dic.items(), key=operator.itemgetter(1))
	boundaries = dict(sorted_x).keys()
	text_legend= process_image(legend)

	plot_list[plotnum].append(title)
	plot_list[plotnum].append(xname)
	plot_list[plotnum].append(yname)
	plot_list[plotnum].append(text_legend.splitlines())
	plot_list[plotnum].append([])
	# loop over the boundaries
	for (b,g,r) in boundaries:
		# create NumPy arrays from the boundaries
		# print "plotting"
		print (b,g,r)
		lower = np.array((0 if (b-40<0) else b-40,0 if (g-40<0) else g-40,0 if (r-40<0) else r-40), dtype = "uint8")
		upper = np.array((255 if (b+40>255) else b+40,255 if (g+40>255) else g+40,255 if (r+40>255) else r+40), dtype = "uint8")
	 	
		mask = cv2.inRange(crop_img, lower, upper)
		output = cv2.bitwise_and(crop_img, crop_img, mask = mask)
		kernel = np.ones((3,3),np.uint8)
		dilation= cv2.dilate(mask,kernel,iterations = 1)
		dilation2= cv2.dilate(dilation,kernel,iterations = 1)
		med=cv2.erode(dilation2,kernel,iterations=1)
		cv2.imwrite("mask.jpg",med)
		count=0		
		inter_pts=[]		
		points=0
		missing=0
		li=[]
	        start_x_plot = 0
	        end_x_plot = crop_img.shape[1]
		start_y_plot=0
		end_y_plot=crop_img.shape[0]
		if(len(li_x)==0):
			li_x.append(0)
		if(len(li_y)==0):
			li_y.append(0)

		#for i in range(start_x_plot,end_x_plot,10):
		for i in range(start_x_plot,end_x_plot,gap):
			# print "loop"
			check=0
			for j in range(start_y_plot,end_y_plot,gap):			
				if (med[j][i]==255):				
					if (not (j>min(li_y) and j<max(li_y)))   and check==0:
						points+=1
						#print(i,j)
						li.append((float((i-x_o)*1.0/dx),float((y_o-j)*1.0/dy)))
						if missing!=0:
							for i in inter_pts:
								points+=1
								a=float((i-x_o)*1.0/dx)
								(x1,y1)=li[len(li)-2]
								(x2,y2)=li[len(li)-1]	
								slope=(y2-y1)*1.0/(x2-x1)
								li.append((a,y2-slope*(x2-a)))
							missing=0
							inter_pts[:] = []
						check=1
			if(check==0):
				if (points==0):
					li.append((float((i-x_o)*1.0/dx),"-"))
				else:
					missing+=1
					inter_pts.append(i)
			
		plot_list[plotnum][4].append(li)
		for i in inter_pts:
			li.append((float((i-x_o)*1.0/dx),"-")) 
		for datapoint in li:
			print(datapoint)
		print(len(li))
	
		# show the images
		# cv2.imwrite("text_legend.jpg",crop_img)


	print dic
	# print plot_list[0]
	# break




pickle.dump(plot_list, open('table.p', 'wb'))
