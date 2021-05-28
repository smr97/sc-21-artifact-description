import torch
import pickle
from random import randint
from random import random

def create_basegraph(N, T, nnz_frame):
    A = []
    for t in range(T):
        index = torch.zeros([nnz_frame,2], dtype=torch.long)
        val = torch.zeros([nnz_frame], dtype=torch.float)
        for i in range(nnz_frame):
            x = randint(0, N-1)
            y = randint(0, N-1)
            if x == y: y = (x+1)%N
            index[i,0] = x
            index[i,1] = y
            #index[i,0] = min(x, y)
            #index[i,1] = max(x, y)
            val[i] = random()
        graph = torch.sparse.FloatTensor(index.transpose(1,0), val, torch.Size([N, N])).coalesce()
        A.append(graph)
    return A

def create_basegraph_same(N, T, nnz_frame):
    A = []
    for t in range(T):
        index = torch.randint(0,N, (nnz_frame,2), dtype=torch.long)
        #index = torch.zeros([nnz_frame,2], dtype=torch.long)
        val = torch.rand(nnz_frame, dtype=torch.float)
        #val = torch.zeros([nnz_frame], dtype=torch.float)
        graph = torch.sparse.FloatTensor(index.transpose(1,0), val, torch.Size([N, N])).coalesce()
        A.append(graph)
    return A
    
def main():
    EF = 5
    N = 10000000
    T = 100
    nnz_frame = EF*N
    A = create_basegraph_same(N, T, nnz_frame)
    path = "sparse_graph_full.pt"
    with open(path, "wb") as f:
        torch.save(N, f)
        torch.save(T, f)
        for i in range(T):
            torch.save(A[i]._indices(), f)
            torch.save(A[i]._values(), f)
    
if __name__ == "__main__":
    main()
