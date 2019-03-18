import numpy as np
import cv2

class Node(object):
    data=[0,0]

    def __init__(self,val,p=0):
        self.data = val
        self.next = p
    def __repr__(self):
        '''
        用来定义Node的字符输出，
        print为输出data
        '''
        return str(self.data)
class LinkList(object):
    def __init__(self):
        self.head = 0
    def __getitem__(self, key):

        if self.is_empty():
            print ('linklist is empty.')
            return

        elif key <0  or key > self.getlength():
            print ('the given key is error')
            return

        else:
            return self.getitem(key)
    def __setitem__(self, key, value):

        if self.is_empty():
            print ('linklist is empty.')
            return

        elif key <0  or key > self.getlength():
            print ('the given key is error')
            return

        else:
            self.delete(key)
            return self.insert(key)
    def initlist(self,data):
        self.head = Node(data[0])
        p = self.head
        for i in data[1:]:
            node = Node(i)
            p.next = node
            p = p.next
    def getlength(self):

        p =  self.head
        length = 0
        while p!=0:
            length+=1
            p = p.next

        return length
    def is_empty(self):

        if self.getlength() ==0:
            return True
        else:
            return False
    def append(self,item):
        q = Node(item)
        if self.head ==0:
            self.head = q
        else:
            p = self.head
            while p.next:
                p = p.next
            p.next = q
    def appendTree(self,treeHead):
            p = self.head
            while p.next:
                p = p.next
            p.next=treeHead
    def getitem(self,index):
        if self.is_empty():
            print ('Linklist is empty.')
            return
        j = 0
        p = self.head
        while p.next!=0 and j <index:
            p = p.next
            j+=1
        if j ==index:
            return p.data

        else:

            print ('target is not exist!')
    def printf(self):
        if self.head==0:
            print("null tree")
        else:
            p=self.head
            while p:
                print(p.data)
                p=p.next
    def delete_val(self, item):
        if self.head==0:
            return None
        p = self.head
        pre=self.head
        while p.next:
            if p.data==item:
                if pre==self.head:
                    pre=p.next
                else:
                    p=p.next
                return
            else:
                pre=p
                p=p.next

        if p.data==item:
            pre.next=0
    def delete(self,index):
        if self.isEmpty():
            exit(0)
        if index < 0 or index > self.getLength() - 1:
            print
            "\rValue Error! Program Exit."
            exit(0)

        i = 0
        p = self.head
        # 遍历找到索引值为 index 的结点
        while p.next:
            pre = p
            p = p.next
            i += 1
            if i == index:
                pre.next = p.next
                p = None
                return 1

        # p的下一个结点为空说明到了最后一个结点, 删除之即可
        pre.next = None
    def getItemList(self):
        item_list=[]
        if self.head==0:
            print('Linklist is empty.')
            return item_list
        p=self.head
        while p:
            item_list.append(p.data)
            p=p.next
        return  item_list
    # def WaveW(self):
    #     # print(wave_end[1]-wave_start[1])
    #     return self.wave_end[1] - self.wave_start[1]


def setTreeTagImage(head,tag_img,val):
    p=head
    while p:
        tag_img[p.data[0]][p.data[1]]=val
        p=p.next


original_img = cv2.imread("F:/1.png")
img_shape= original_img.shape
ls=[]
w=img_shape[0]
h=img_shape[1]
tag_img=np.ones((w,h)).astype(np.int16)
tag_img=tag_img*-1
# label_set=np.ones((w,h)).astype(np.int8)
# label_set=tag_img*-1
# label_set_count=0
# canny(): 边缘检测
img1 = cv2.GaussianBlur(original_img, (3, 3), 0)
gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
_,gray=cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
cv2.imshow("gray", gray)    #255白
list_lslink=[]
type_num=0
for i in range(w) :
    if gray[i][0]==0:  #0黑
        link_list=LinkList()
        tag_img[i][0]=type_num
        #label_set[i][0]=label_set_count
        link_list.append(([i,0]))
        list_lslink.append(link_list)
        type_num+=1

        #label_set_count+=1
for j in range(1,h) :
    if gray[0][j]==0: #0黑
        link_list=LinkList()
        tag_img[0][j] = type_num
        #label_set[0][j] = label_set_count
        link_list.append(([0,j]))
        list_lslink.append(link_list)
        type_num+=1

        #label_set_count += 1
count=0
for j in range(1,h):
    for i in range(1,w):
        type_list = []
        posi_list=[]
        min_index=-1

        if gray[i][j]==0:
            if gray[i-1][j]==0:
                type_list.append(tag_img[i-1][j])   #正上
                posi_list.append([i - 1, j])
            if gray[i - 1][j - 1] == 0:
                type_list.append(tag_img[i - 1][j - 1])  #左上
                posi_list.append([i - 1, j - 1])
            if gray[i ][j - 1] == 0:
                type_list.append(tag_img[i][j - 1])   #左前
                posi_list.append([i, j - 1])
            if i<w-1:
                if gray[i +1][j-1] == 0 :
                    type_list.append(tag_img[i+1][j-1])   #左下
                    posi_list.append([i+1,j-1])
            if len(type_list) > 0:
                min_index= type_list.index(min(type_list))
                for c in range(len(type_list)):
                    if c!=min_index:
                        #print(type_list[min_index])
                        if type_list[min_index]!=type_list[c] and list_lslink[type_list[c]]!=None:  ##当领域相同时 None一次之后其余变空
                            # if list_lslink[type_list[c]]==None:
                            #     print(2222)
                            #     print(1111)
                            setTreeTagImage(list_lslink[type_list[c]].head,tag_img,type_list[min_index])
                            list_lslink[type_list[min_index]].appendTree(list_lslink[type_list[c]].head)
                            list_lslink[type_list[c]]=None
                #对前链表所有minlabel进行设置
                tag_img[i][j] = type_list[min_index]
                list_lslink[type_list[min_index]].append(([i, j]))
            else:
                link_list = LinkList()
                tag_img[i][j] = type_num
                link_list.append([i, j])
                list_lslink.append(link_list)
                type_num += 1
pic = np.zeros((w, h, 3), np.uint8)
#print(len(list_lslink))
result_list=[]
for list in list_lslink:
    if list:
        result_list.append(list)
print(len(result_list))
for item in result_list:
#     pic[item[0]][item[1]]=(0,0,255)
    p=item.head
    while p:
        pic[p.data[0]][p.data[1]] = (0, 0, 255)
        p=p.next
cv2.imshow("pic", pic)
cv2.waitKey(0)