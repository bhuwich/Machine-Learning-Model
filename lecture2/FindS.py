class FindS:
    def __init__(self):
        self.h = []
    def train(self, x,t):
        if len(x) >0:
            self.h = [0] * len(x[0]) # len(x[0]) = จำนวนคอลัมน์

        for i in range(len(x)): #len(x) = จำนวนของแถว
            if t[i] == 'Yes':
                for j in range(len(x[i])):
                    if self.h[j] ==0:
                        self.h[j] = x[i][j]
                    else:
                        if self.h[j] != x[i][j]:
                            self.h[j] = '?'


    def predict(self,x):
        output = ['Yes'] * len(x)
        for i in range(len(x)):
            for j in range(len(self.h)):
                if self.h[j] != '?':
                    if self.h[j] != x[i][j]:
                        output[i] = 'No'
                        break
        return output
if __name__ == '__main__':
    X = [['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'],
         ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'],
         ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'],
         ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change']
        ]
    T = ['Yes', 'Yes', 'No', 'Yes']
    find_s = FindS()
    find_s.train(X, T)
    list_output = find_s.predict([['Sunny','Warm','Normal','Strong','Cool','Change'],
                                 ['Sunny','Cold','Normal','Strong','Warm','Change']
                                 ])
    print(find_s.h)
    print(list_output)