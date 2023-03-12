import numpy as np
class KNN:
    def __init__(self, k):
        self.k = k
        self.x = []
        self.y = []
    
    def distance(self, p1, p2):
        #p1 = [x1, x2, x3, x4]
        #p2 = [y1, y2, y3, y4]
        #ใช้วิธีการหา euclidean distance
        #distance = sqrt( (x1-x2)^2 + (x2-y2)^2 + (x3-y3)^2 + (x4-y4)^2)
        p1 =  np.array(p1)
        p2 = np.array(p2)
        
        if len(p1) != len(p2):
            raise ValueError("invalid input p1 and p2 !!!")
        return np.sqrt( sum((p1- p2)**2 ))
    
    def train(self,x,y):
        self.x = np.array(x)
        self.y = np.array(y)

    def predict(self, x):
        #เช่น x = [[x1,x2,x3,x4],[y1,y2,y3,y4],[z1,z2,z3,z4]]
        #output = [Iris-setosa,Iris-versicolor,Iris-versicolor] ที่ต้องการส้งออกจากฟังก์ชัน
        output = []
        for i in range(len(x)):
            #ทำนาย x[i]ทีละตัว
            #เมื่อทำนาย x[i] เสร็จจะถูก append เข้าไปใน output

            #คำนวณ distance จาก x[i]ไปที่ x ทุกตัวใน data train
            list_distance = []
            for j in range(len(self.x)):
                list_distance.append(self.distance(x[i],self.x[j]))
            list_distance = np.array(list_distance)

            index_sort = np.argsort(list_distance)
            yp_ = self.y[index_sort[:self.k]]
            values , counts = np.unique(yp_, return_counts=True)
            
            output.append(values[np.argmax(counts)])
        
        return output

if __name__ == "__main__":
    from datasetHandler import loadIrisDataset, splitTrainTest
    x, y = loadIrisDataset()
    x_train, y_train, x_test, y_test = splitTrainTest(x,y)
    knn = KNN(50)
    knn.train(x_train, y_train)
    output = knn.predict([ [6.1, 2.6, 5.6, 1.4], [5.4, 3.0, 4.5 ,1.5]])
    print(output)