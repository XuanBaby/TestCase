import torch


def del_tensor_ele(tensorA,tensorB):
    tensorB = torch.sort(tensorB).values
    len_A = len(tensorA)
    for index in tensorB:
        tensorA_1 = tensorA[0:index - (len_A - len(tensorA))]
        tensorA_2 = tensorA[(index - (len_A - len(tensorA)))+1:]
        tensorA = torch.cat((tensorA_1,tensorA_2))
        print(tensorA)
    return tensorA

A = torch.tensor([5, 4, 0, 3, 1, 2])
B = torch.tensor([5, 1, 2, 3, 4])

C = del_tensor_ele(A,B)
print(C)
