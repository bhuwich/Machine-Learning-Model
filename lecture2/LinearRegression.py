import numpy as np
class LinearRegression:
    def __init__(self):
        self.m = 0
        self.c = 0
    def train(self, x, y):
        # x y เป็นลิส 1 มิติ
        X = np.array(x)
        Y = np.array(y)

        sum_x = np.sum(X)
        sum_y = np.sum(Y)
        sum_x2 = np.sum(X**2)
        sum_xy = np.sum(np.array(X*Y, dtype=np.int64))
        N = len(X)

        #A*X = C
        # X = inv(A) * C

        A = np.array([[sum_x, N],
                    [sum_x2, sum_x]])
        C = np.array([[sum_y],
                    [sum_xy]])
        P = np.dot(np.linalg.pinv(A), C )
        self.m = P[0,0]
        self.c = P[1,0]
    def predict(self, x):
        #ลิสของข้อมูลที่ต้องการจะทำนาย
        return self.m*np.array(x) + self.c

if __name__ == '__main__':
    Y = [199000, 245000, 319000,240000, 312000, 279000, 310000, 405000, 405000, 324000]
    X = [1100, 1400, 1425, 1550, 1600, 1700, 1700, 2350, 2350, 2450]
    linear_regression = LinearRegression()
    linear_regression.train(X, Y)
    print(f"m = {linear_regression.m},c = {linear_regression.c}")
    output = linear_regression.predict([2000])
    print(output)