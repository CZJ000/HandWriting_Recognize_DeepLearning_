import numpy as np
import cv2

from enum import Enum


class MergeType(Enum):
    left=0
    right=1
    none=2
class Wave:
    '所有员工的基类'
    wave_start = [0,0]
    wave_top = [0,0]
    wave_end = [0,0]
    def __init__(self, wave_start, wave_top,wave_end):
        self.wave_start = wave_start
        self.wave_top = wave_top
        self.wave_end = wave_end
    def WaveH(self):
        return self.wave_top[0]-self.wave_start[0] if self.wave_start[0]<self.wave_end[0] else self.wave_top[0]-self.wave_end[0]
    def WaveW(self):
        #print(wave_end[1]-wave_start[1])
        return self.wave_end[1]-self.wave_start[1]
original_img = cv2.imread("E:/16.png")
img_shape= original_img.shape
ls=[]
w=img_shape[0]
h=img_shape[1]
gray=cv2.cvtColor(original_img,cv2.COLOR_BGR2GRAY)
#二值化处理
_,gray=cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)
cv2.imshow("G", gray)
pic = np.zeros((w, h, 3), np.uint8)
#画出下轮廓

for y in range(0,h):
    lowPoint = [0, 0]
    exist=0
    for x in range(0, w):
        if gray[x, y] == 255:   #255白色
                if x>lowPoint[0]:
                     exist=1
                     pic[lowPoint[0], lowPoint[1]] =(255, 255, 255)
                     lowPoint=[x,y]
                else:
                     pic[x,y]=(255, 255, 255)
                #out[lowPoint[0],lowPoint[1]]=255
        else:
            pic[x, y] = (255, 255, 255)
    if exist:
        ls.append(lowPoint)
        pic[lowPoint[0], lowPoint[1]] = (0,0,255)

cv2.imshow('origi', pic)
cv2.waitKey(0)
#得到下轮廓波峰和波谷
wave_list=[]
num=0
wave_start=ls[0]
wave_start_h=ls[0][0]
wave_top_h=wave_start_h
wave_top=[]
wave_end_h=0
wave_end=[]
top_find=0
for i in range(0,len(ls)):
     if top_find==1:
         if ls[i][0]<=wave_end_h:
             wave_end_h = ls[i][0]
             wave_end = ls[i]
         else:
            num += 1
            wave_list.append(Wave(wave_start,wave_top,wave_end))
            wave_start = wave_end
            wave_start_h=wave_end_h
            wave_top=[]
            wave_end=[]
            top_find=0
            wave_top_h = wave_start_h
         if i == len(ls)-1:
             num += 1
             wave_list.append(Wave(wave_start, wave_top, wave_end))

     else:
        if ls[i][0]>=wave_top_h :
            wave_top=ls[i]
            wave_top_h=ls[i][0]
        else :
            top_find=1
            wave_end_h=ls[i][0]
            wave_end = ls[i]

print('num:'+str(num))
avg_h=0
avg_w=0
for i in wave_list:
    avg_h+=i.WaveH()
    avg_w+=i.WaveW()
    print("waveH:"+str(i.WaveH()))
avg_h/=num
avg_w/=num

wave_list_result=[]

list_length=len(wave_list)

#波峰筛选
wave_list_result=[]
count=1
mergeType= MergeType.none

merge_wave=wave_list[0]
while count<list_length:
    # new_wave=Wave()
    if merge_wave.WaveH()<avg_h*0.1:
        if len(wave_list_result)>0 :
            if count==list_length-1:
                mergeType = MergeType.left
            else:
                last_wave=wave_list_result[len(wave_list_result)-1]
                r_wave=wave_list[count]
                if last_wave.WaveH()<r_wave.WaveH():
                    mergeType = MergeType.left
                else:
                    mergeType = MergeType.right
        else:
            mergeType = MergeType.right
    elif  merge_wave.WaveH()<avg_w*0.3:
        if len(wave_list_result)>0 :
            if count==list_length-1:
                mergeType = MergeType.left
            else:
                last_wave=wave_list_result[len(wave_list_result)-1]
                r_wave=wave_list[count]
                if last_wave.WaveW()<r_wave.WaveW():
                    mergeType = MergeType.left
                else:
                    mergeType = MergeType.right
        else:
            mergeType = MergeType.right
    elif  abs(merge_wave.WaveH()-wave_list[count].WaveH())>0.75*max(merge_wave.WaveH(),wave_list[count].WaveH()):
        mergeType = MergeType.right
    else:
        print("none")
        wave_list_result.append(merge_wave)
        merge_wave=wave_list[count]
        count+=1
        mergeType = MergeType.none
    if mergeType==MergeType.left:
        # if len(wave_list_result)>0:
        index=len(wave_list_result)-1
        last_wave=wave_list_result[index]
        l_top=last_wave.WaveH()
        m_top=merge_wave.WaveH()
        last_wave.wave_top=last_wave.wave_top if l_top>r_top else merge_wave.wave_top
        last_wave.wave_end=merge_wave.wave_end
        count += 1
        print("left")
    elif mergeType==MergeType.right:
        m_top=merge_wave.WaveH()

        r_top=wave_list[count].WaveH()

        merge_wave.wave_end=wave_list[count].wave_end
        merge_wave.wave_top=merge_wave.wave_top if m_top>r_top else wave_list[count].wave_top
        count += 1
        print("right")

wave_list_result.append(merge_wave)



count=0
cut_posi=[]
cut_left=0
cut_right=0
while count<len(wave_list_result)-1:
    cut_left = cut_right
    l_wave=wave_list_result[count]
    r_wave = wave_list_result[count+1]
    start_posi=[0,0]
    if l_wave.wave_top[0]>r_wave.wave_top[0]:
        start_posi = r_wave.wave_top

        find_black=0
        for y in reversed(range(r_wave.wave_top[1]+1)):
            if find_black==1:
                if (gray[start_posi[0]][y]) == 255:
                    start_posi=[start_posi[0],y+1]
                    break
            else:
                if(gray[start_posi[0]][y])==0:   #255 白色
                    find_black=1
                    start_posi=[start_posi[0],y]
    else:
        start_posi = l_wave.wave_top
        y=start_posi[1]
        while y<=h:
            if (gray[start_posi[0]][y]) == 255: #255
                start_posi = [start_posi[0], y]
                break;
            y+=1
    refer_Y=wave_list_result[count+1].wave_start[1]
    start_x=start_posi[0]
    start_y=start_posi[1]
    print("start_y"+str(start_y))
    print("refer_Y"+str(refer_Y))
    while start_x>0 and start_y>0 and start_y<h:
        if start_y==refer_Y: break
        if start_y<refer_Y:    #右优先
            if  gray[start_x-1][start_y+1] ==0 :#255 右上白
                start_x = start_x - 1
                start_y = start_y + 1
            elif gray[start_x-1][start_y] ==0 :#255 上白
                start_x=start_x-1
            elif gray[start_x - 1][start_y - 1] == 0:#255 左上白
                start_x = start_x - 1
                start_y = start_y - 1
            elif gray[start_x ][start_y + 1] == 0:#255 右白
                start_y = start_y + 1
            elif gray[start_x ][start_y - 1] == 0:#255 左白
                start_y = start_y + 1
            else:
                start_x = start_x-1
        else:                   #左优先
            if gray[start_x - 1][start_y - 1] == 0:  # 255 左上白
                start_x = start_x - 1
                start_y = start_y - 1
            elif  gray[start_x-1][start_y] ==0 :#255 下白
                start_x=start_x-1
            elif gray[start_x-1][start_y+1] ==0 :#255 右上白
                start_x = start_x-1
                start_y=start_y+1
            elif gray[start_x][start_y - 1] == 0:  # 255 左白
                start_y = start_y - 1
            elif gray[start_x ][start_y + 1] == 0:#255 右白
                # if start_y-1==refer_Y:
                #     start_y=refer_Y
                #     break
                start_y = start_y - 1
            else:
                start_x = start_x-1
    end_posi=[start_x,start_y]
    cut_right=start_y
    print("end_posi"+str(end_posi))
    cut=gray[0:w,cut_left:cut_right]
    result_pic=cv2.resize(cut, (28, 28), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("pic", result_pic)
    cv2.imwrite("E:/" + str(count) + "__.png", result_pic)
    for i in range(w):
        gray[i][start_y]=255
    count+=1
cut=gray[0:w,cut_right:h]
result_pic=cv2.resize(cut, (28, 28), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("pic", result_pic)
cv2.imwrite("E:/" + str(count) + "__.png", result_pic)
cv2.imshow('cut', gray)
cv2.waitKey(0)
# avg_w+=i.WaveW()
# print(i.WaveH())
#
#
# print("avh:"+str(avg_h))
# print("avw:"+str(avg_w))













# print(w,h)
# #np.zeros((width, height, 3), np.uint8)
# pic=np.zeros((w, h, 3),np.int8)
#
#
#
# for x in range(0,w):
#     for y in range(0, h):
#         b = (0,0,255)
#         pic[x,y] = b
#
# cv2.imshow('chuizhi', pic)


# for x in range(0,w):
#     lowPoint = [0, 0]
#     for y in range(0, h):
#         # if canny[x, y] == 0:
#         #     if x>lowPoint[0]:
#         #         lowPoint=[x,y]
#         # else:
#         #     pic[x,y]=(0,0,0)
#         #out[lowPoint[0],lowPoint[1]]=255
#         b = (0,0,255)
#         pic[x,y] = b
# cv2.imshow('img', pic)


# for j in range(h):
#     lowPoint=[0,0]
#     for i in range(w):
#         if canny[i,j] ==0:
#             lowPoint=[i,j]
#     out[lowPoint[0],lowPoint[1]]=255



#contours,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# print("len(contours)=%d"%(len(contours)))
# = len(contours)



#cv2.drawContours(original_img, contours, -1, (0, 0, 255), 1)

# for c in contours:
#     print(c[0][0])
#     ls.append(list(c[0][0]))
# sort_ls=sorted(ls)




# #
# maxArea=-9999
# maxindex=0
# docCnt = None
# # 确保至少找到一个轮廓
# if len(contours) > 0:
#     # 按轮廓大小降序排列
#     cnts = sorted(contours, key=cv2.contourArea, reverse=True)
#     for c in cnts:
#         # 近似轮廓
#         peri = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#         # 如果我们的近似轮廓有四个点，则确定找到了纸
#         if len(approx) == 4:
#             docCnt = approx
#             break
#
#
