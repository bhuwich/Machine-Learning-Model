import numpy as np

tmp = np.arange(-1,1.1,0.1)
m_range = np.array([tmp for _ in range(len(tmp))])
c_range = np.transpose(m_range)