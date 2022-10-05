import torch
import numpy as np
class problem:
    def __init__(self, max_min, A, C, b, basis, ss=None):
        self.max_min = max_min
        self.A_matrix = A
        self.C_matrix = C
        self.b_vector = b
        self.basis_index = basis
        self.dimension = list(self.b_vector.size())[0]
        if ss != None:
            self.I_matrix = ss
        else:
            self.I_matrix = torch.eye(self.dimension)
        self.Concatenation()

    def Concatenation(self):
        ''' 연산을 위한 세팅 과정'''
        self.Z_matrix = torch.zeros(*list(self.b_vector.size()))
        self.A_matrix = torch.cat((self.A_matrix,self.I_matrix),1)
        self.C_matrix = torch.cat((self.C_matrix,self.Z_matrix),0)
        self.trials = list(set([i for i in range(0,*list(self.C_matrix.size()))]))

    def solve(self):
        for i in (self.trials):
            inverse_B = torch.inverse(self.A_matrix[:,self.basis_index])
            A_i = self.A_matrix[:,i]
            flag = self.C_matrix[i] - self.C_matrix[self.basis_index].matmul(inverse_B).matmul(A_i)
            #print(inverse_B) # 각 시도 때 inverse B 확인 시 #해제
            #print(flag) # 각 시도 때 rj 값 확인 시 # 해제
            if self.max_min == 0 and flag.item() < 0:
                self.step_2(inverse_B, A_i, i)
            if self.max_min == 1 and flag.item() > 0:
                self.step_2(inverse_B, A_i, i)
        inverse_B = torch.inverse(self.A_matrix[:,self.basis_index])
        print(f'최종 basis index는 {self.basis_index}')
        print(f'최적 X는 {inverse_B.matmul(self.b_vector)}')
        print(f'최적 Z는 {self.C_matrix[self.basis_index].matmul(inverse_B.matmul(self.b_vector))}')

    def step_2(self, inverse_B, A_i, i):
        a = list(inverse_B.matmul(self.b_vector))
        b = list(inverse_B.matmul(A_i))
        c = [float('inf') for i in range(self.dimension)]
        for j in range(self.dimension):
            try:
                if a[j] / b[j] > 0:
                    c[j] = a[j] / b[j]
            except:
                pass
        self.basis_index[c.index(min(c))] = i
        self.basis_index.sort()
        #print(f'basis : {self.basis_index}') #basis 변환 시에 basis 값 확인 시 # 삭제

if __name__ == '__main__':
    max_min = 0 #minimize면 0, maximize면 1로 설정
    C_matrix = torch.FloatTensor(np.array([-1, -2, 1, -1, -4, 2])) #목적식
    A_matrix = torch.FloatTensor(np.array([[1, 1, 1, 1, 1, 1], [2, -1, -2, 1, 0, 0], [0, 0, 1, 1, 2, 1]])) #제약식 행단위 입력
    b_vector = torch.FloatTensor(np.array([6, 4, 4]))
    basis_index = [6, 7, 8] #0부터 시작함(실제로 x1, x2, x3 basis로 사용시 0, 1, 2 입력

    p1 = problem(max_min,A_matrix,C_matrix,b_vector,basis_index)
    #surplus 변수나 slack 변수 세팅 따로 필요한 경우 윗 줄 대신 아래 두 줄 사용, 기본은 identity matrix로 되어있음.
    #surplus_slack = torch.FloatTensor(np.array([[1,0,0],[0,1,0],[0,0,1]]))
    #p1 = problem(max_min,A_matrix,C_matrix,b_vector,basis_index,surplus_slack)
    p1.solve()
