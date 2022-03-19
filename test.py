import torch

if __name__ == '__main__':

    start_range = torch.arange(10, dtype=torch.float).expand(10, 10)
    # print(start_range)
    tmp = torch.tensor([2, 3, 5, 3, 4, 4, 9, 3, 8, 1]).view(10, 1)
    test = start_range - tmp
    sigma = 0.3
    test =torch.exp(-((start_range - tmp) ** 2) / (2 * sigma ** 2)).view(10, 10).permute(1, 0)
    feature = torch.rand(5, 10)
    print(feature)
    # print(tmp)
    f_f = torch.mm(feature, test)
    print(f_f)
    # print(start_range)
