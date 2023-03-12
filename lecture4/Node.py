import numpy as np
from entropy import entropy
class Node:
    def __init__(self,name =""):
        self.name = name
        self.children = [] # สมาชิกข้างในจะเป็น Node
        self.branch = [] #ลิสของสตริง (ป้ายกำกับของ branch)
        # children และ branch จะต้องมีสมาชิกเท่ากัน
        # และสัมพันธ์กันตาม index
        self.major_class = ""
        self.threshold_entropy = 0 # entropy ที่มากที่สุดสำหรับการเปลี่ยน node เป็น Leaf
        self.threshold_sample = 0 # จำนวณ sample ที่มากที่สุดสำหรับการเปลี่ยน node เป็น leaf
    
    # Node สามารถกลายเป็น Lead ได้ ถ้า Node ไม่มี Children
    # เมื่อ Node เป็น Leaf ชื่อของNode จะเป็นชื่อคลาส

    def isLeaf(self):
        if len(self.children) == 0:
            return True
        return False

    def isNode(self):
        return not(self.isLeaf())

    def train(self,x,t,f=None, is_numeric=None):
        x, t, f = self.prepareData(x, t, f, is_numeric)
        # สร้าง major class สำหรับ Node
        # เราจะเลือก class ที่มีจำนวนสูงที่สุดใน t
        values,count = np.unique(t, return_counts=True)
        self.major_class = values[np.argmax(count)]

        #คำนวณ entropy ของข้อมูลก่อน
        hs =entropy(t)

        if hs > self.threshold_entropy and len(t) > self.threshold_sample:

            best_index,best_f = self.bestSplit(x,t,f)
            if len(np.unique(x[:, best_index])) == 1:
                self.name = self.major_class
                self.branch = []
                self.children = []
                return 
            # ตั้งชื่อให้กับ Node
            self.name = best_f
            #print(f"Create node : {self.name}")
            # กำหนด branch 
            self.branch = np.unique(x[:, best_index]).tolist()
            # เช่น best_index =0, self.branch= ['sunny','overcast','rain']
            
            #การสร้าง children และส่งข้อมูลไปให้ไป children train ต่อ
            
            for v in self.branch:
                ind_v = x[:,best_index] == v
                x_cut = x[ind_v,:]
                t_cut = t[ind_v]
                child_node = Node()
                child_node.threshold_entropy = self.threshold_entropy
                child_node.threshold_sample = self.threshold_sample
                child_node.train(x_cut,t_cut,f)
                self.children.append(child_node)
        else:
            #กำหนด node เป็น ใบ
            self.name = self.major_class
            self.branch = []
            self.children = []
    def bestSplit(self,x,t,f):
        x = np.array(x)
        t = np.array(t)
        f = np.array(f)
        num, num_attr = x.shape #num = จำนวน sample, num_attr = จำนวน feature
        hs = entropy(t)
        grs = hs *np.ones(num_attr)
        for i in range(num_attr):
            unique_v = np.unique(x[:,i]) #กิ่งของ feature ที่ i
            tmp_sum = 0
            si = 0
            for v in unique_v:
                ind_v = x[:,i] == v
                t_v = t[ind_v]
                hv = entropy(t_v)
                tmp = (np.sum(ind_v)/num)*hv
                tmp_sum += tmp
                si += (np.sum(ind_v)/num)*np.log10(np.abs((np.sum(ind_v/num))-0.001)) #ใส่ -0.001 กัน error
            grs[i] -= tmp_sum
            grs[i] /= -si-0.001
        best_index = np.argmax(grs)
        best_f = f[best_index]
        return best_index,best_f

    def predict(self, x, f=None):
        # predict จะทำการ predict หลาย sample
        # เช่น x = [['hot','circle','blue'], ['cool','square','green']] = 2 sample
        # นอกจากนั้น user ต้องกำหนด ชื่อของ Feauture เข้ามาด้วย = f
        # f = ['Temperature','Shape','Color'] ถ้าใส่เข้ามา
        # f = ['0','1','2'] กรณีไม่ใส่แล้วโปรแกรมตั้งเอง
        
        if f is None:
            f = [str(j) for j in range(x.shape(1))]
        f = np.array(f)
        
        output = []
        for i in range(len(x)):
            output.append(self._predict(x[i],f)) # self._predict จะทำการ predict ทีละ sample
        return output
    def _predict(self, x_, f):
        if self.isNode():
            x_ = np.array(x_)
            f = np.array(f)
            
            idf = f.tolist().index(self.getName()) #f == self.name
            if (self.isNumeric()):
                value = float(x_[idf])
                if value >= self.getValue():
                    idv = self.branch.index("True")
                else:
                    idv = self.branch.index("False")
                return self.children[idv]._predict(x_,f)
            else:
                if x_[idf] in self.branch:
                    idv = self.branch.index(x_[idf])
                    return self.children[idv]._predict(x_,f)
                else:
                    return self.major_class

        elif self.isLeaf():
            return self.name
    
    
    
    #เนื่องจากเป็น obj ref copynode ไว้เพื่ออันเดิมจะได้ไม่เปลี่ยน 
    def copyNode(self, node, node_des):
        node_des.name = node.name
        node_des.children = list(node.children)
        node_des.branch = list(node.branch)
        node_des.major_class = node.major_class
        node_des.threshold_entropy = node.threshold_entropy
        node_des.threshold_sample = node.threshold_sample

    def convertNode2Leaf(self, node):
        node.name = node.major_class
        node.children = []
        node.branch = []

    def evaluate(self, target, detect):
        n_correct = 0
        for i in range(len(target)):
            if target[i] == detect[i]:
                n_correct += 1
        return n_correct/len(target)

    def appendListNode(self, node, list_node):
        for i in range(len(node.children)):
            if node.children[i].isNode():
                list_node.append(node.children[i])
                self.appendListNode(node.children[i], list_node)

    def createListNode(self, node):
        list_node = []
        self.appendListNode(node, list_node)
        return list_node

    def postPruning(self, x_test, t_test, f=None):
        #convert to numpy array
        x_test = np.array(x_test)
        t_test = np.array(t_test)
        if f is None:
            f = [str(i) for i in range(x_test.shape[1])]
        f = np.array(f)

        #ต้องหา List ของ Node ทุก Node ใน children
        list_node = self.createListNode(self)

        #เริ่มทำการ pruning 
        output_test = self.predict(x_test, f)
        prev_acc = self.evaluate(t_test, output_test)
        print(f"--------Before post pruning -> acc = {prev_acc*100:.2f}% ---\n")
        iteration =0
        while True:
            list_acc = []
            for i in range(len(list_node)):
                #copy node เพื่อเก็บข้อมูลของ nodeไว้
                node_tmp = Node()
                self.copyNode(list_node[i], node_tmp)

                #เปลี่ยน list_node เป็นใบ
                self.convertNode2Leaf(list_node[i])

                #วัดผล
                output_test = self.predict(x_test, f)
                list_acc.append(self.evaluate(t_test, output_test))

                #แทน node เดิมกลับเข้าไป
                self.copyNode(node_tmp, list_node[i])

            # หา index ที่ทำให้ acc สูงสุด
            if len(list_acc) ==0:
                print(f"[{iteration}] break because list_node is empty !!!")
                break
            max_ind = list_acc.index(max(list_acc))

            #check เงื่อนไขว่า acc ตกลงหรือไม่
            if list_acc[max_ind] < prev_acc:
                print(f"({iteration}) break !!!, prev_acc = {prev_acc*100:.2f}%, cur_acc = {list_acc[max_ind]*100:.2f}%")
                break
            else:
                #เปลี่ยน node ที่ max_ind ไปเป็น leaf
                print(f"({iteration}) pruning node: {list_node[max_ind].name}, prev_acc = {prev_acc*100:.2f}%, curr_acc= {list_acc[max_ind]*100:.2f}%")
                self.convertNode2Leaf(list_node[max_ind])
                list_node = self.createListNode(self)
                prev_acc = list_acc[max_ind]
            iteration +=1

    def isNumeric(self):
        return len(self.name.split(" >= ")) == 2 and " ?" in self.name
    def getName(self):
        if self.isNumeric():
            return self.name.split(" >= ")[0]
        return self.name
    def getValue(self):
        if self.isNumeric():
            return float(self.name.split(" >= ")[1].split(" ")[0])
        return 0
    
    def convertNumericFeature(self, x_, t, f_):
        # f_ = "Temperature"
        # x_ = [40,48,60,72,80,90]
        # t = [No,No,Yes,Yes,Yes,No]
        
        
        #ตัวอย่าง output
        # x = [[False, False], [False,False], [True,False], [True,False],[True,False],[True,True]]
        # f = ["Temperature >=54 ?","Tempurature >=85 ?"]
        # values = []
        #x_ = np.array([float(x_[i]) for i in range(len(x_))])
        # for i in range(len(t)-1):
        #     if t[i] != t[i+1]:
        #         values.append(x_[i] + x_[i+1] /2)
        x_ = np.array([float(x_[i]) for i in range(len(x_))])
        #หา values ทั้งหมด
        values = np.array([(x_[i]+x_[i+1])/2 for i in range(len(t)-1) if t[i] != t[i+1]])#values= [54,85]
        f = np.array([f"{f_} >= {values[i]} ?" for i in range(len(values))]) # f = ["Tempurature >=54 ?","Tempurature >= 85"]

        x = [[str(x_[i] >= values[j]) for j in range(len(values))] for i in range(len(t))]
            # tmp = []
            # for j in range(len(values)):
            #     if x_value >= values[j]:
            #         tmp.append("True")
            #     else:
            #         tmp.append("False")
        return x,f
    def prepareData(self, x, t, f=None, is_numeric =None):
        #convert to numpy array 
        x = np.array(x)
        t = np.array(t)

        # ในกรณีที่ไม่ใส่ Numeric เข้ามาจะให้ feature ทุกตัวเป็น nominal ทั้งหมด
        if is_numeric is None:
            is_numeric = [False for _ in range(len(f))]
        is_numeric = np.array(is_numeric)
        # ในกรณีที่ไม่ใส่ f เข้ามาจะทำให้เป็นเลข 0,1
        if f is None:
            f = [str(i) for i in range(x.shape[1])]
        f = np.array(f)
        
        for i in range(len(is_numeric)):
            if is_numeric[i]:
                #convert x[:,i] to nominal
                x_tmp, f_tmp = self.convertNumericFeature(x[:,i],t,f[i])
                x= np.append(x, x_tmp, axis=1)
                f = np.append(f, f_tmp, axis=0)
                is_numeric = np.append(is_numeric, [False]*len(f_tmp), axis=0)
        x = x[:, is_numeric ==False]
        f = f[is_numeric ==False]
        return x,t,f
if __name__ == '__main__':
    node_0 = Node("Color")
    node_1 = Node("Shape")
    node_2 = Node("Shape")
    node_3 = Node("-")
    node_4 = Node("+")
    node_5 = Node("+")
    node_6 = Node("+")
    node_7 = Node("+")
    node_8 = Node("-")

    node_0.children = [node_1, node_6, node_2]
    node_0.branch = ['Blue','Red', 'Green']

    node_1.children = [node_3, node_4,node_5]
    node_1.branch = ['Triangle','Square','Circle']

    node_2.children = [node_7,node_8]
    node_2.branch = ['Square','Circle']

    # โดยปกติ node_0 จะเป็น root node
    output = node_0.predict(
        [['Blue','Circle'],
        ['Red','Square'],
        ['Green','Circle']],
        f= ['Color','Shape']
    )
    print(output)

    from DecisionTree import DecisionTree
    tree = DecisionTree(root_node=node_0)
    tree.show()

    
        