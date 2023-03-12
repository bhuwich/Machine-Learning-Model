import numpy as np
class ListThenElim:
    def __init__(self):
        self.h = []
        self.t = []
    def train(self,x ,t):
        H, T = self.listPosibleH(x, t)

        ind_remove = []
        for i in range(len(H)):
            for j in range(len(x)):
                # ตรวจสอบ x[j] ตรงกับ H[i] มั้ย
                # ถ้าตรงกัน ให้ไปดูว่า t[j] ตรงกับ T[i] มั้ย
                # ถ้า t[j] ไม่ตรงกับ T[i] ให้ ind_remove append i เข้าไป
                # x[j] = ['A','A','B','A']
                # H[i] = ['A','B,'B','A']
                # flag = [True,False,True,False]
                flag = [x[j][k]== H[i][k] for k in range(len(H[i]))]
                if sum(flag) == len(H[i]) and t[j] != T[i]:
                    ind_remove.append(i)
        self.h = H
        self.t = T
        ind_valid = np.setdiff1d(np.arange(len(self.h)), ind_remove)
        self.h = np.array(self.h)[ind_valid]
        self.h = self.h.tolist()
        self.t = np.array(self.t)[ind_valid]
        self.t = self.t.tolist()
            
    def listPosibleH(self, X, T):
        X = np.array(X)
        T = np.array(T)
        n = X.shape[1]
        A = []
        for i in range(n):
            A.append(sorted(list(set(X[:,i]))))
        
        H = []
        t = []
        i = 1

        idx_data = [0] * n
        while True:
            h = []
            for j in range(n):
                h.append(A[j][idx_data[j]])
            
            for tt in np.unique(T):
                H.append(h)
                t.append(tt)
                i += 1
            
            idx_data[-1] += 1
            letter_index = n-1
            while idx_data[letter_index] > len(A[letter_index]) - 1:
                idx_data[letter_index] = 0
                letter_index -= 1
                if letter_index < 0:
                    return H, t
                idx_data[letter_index] += 1
    
    def predict(self, x):
        output = ['Yes']*len(x)
        for j in range(len(x)):
            number_yes = 0
            number_no = 0

            # นับจำนวน
            # เอา x[j] ไปเทียบกับ self.h ทุกตัว แล้วดูว่าตรงกับ self.h แถวที่เท่าไหร่
            # ถ้า เกิดมันตรง ให้ดูว่า self.t ที่แถวนั้นเป็นอะไร
            for i in range(len(self.h)):
                # self.h[i] ตรงกับ x[j] มั้ย?
                # x[j] = ['A',?','B','A']
                # H[i] = ['A','B,'B','A']
                # flag = [True,True,True,False]
                flag = [(x[j][k]== self.h[i][k]) or (x[j][k] == '?') for k in range(len(self.h[i]))]
                if sum(flag) == len(self.h[i]):
                    if self.t[i] == 'Yes':
                        number_yes += 1
                    else:
                        number_no += 1        

        
            
            # ตัดสินใจ
            if number_yes < number_no:
                output[j] = 'No'
        return output


if __name__ == '__main__':
    x = [['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change']
    ]
    t = ['Yes', 'Yes', 'No', 'Yes']
    list_then_elim = ListThenElim()
    list_then_elim.train(x,t)
    h = list_then_elim.h
    t = list_then_elim.t
    for i in range(len(h)):
        print(i+1, h[i], t[i])
    list_output = list_then_elim.predict([['Sunny','Warm','Normal','Strong','Cool','Change'],
                                 ['Sunny','Cold','Normal','Strong','Warm','Change']
                                 ])
    print(list_output)