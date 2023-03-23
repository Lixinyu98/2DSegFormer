import torch
import math
import torch.nn.functional as F

def normpdf(x, std):
    var = std**2
    denom = (2*math.pi*var)**.5
    return torch.exp(-(x)**2/(2*var))/denom

def pdist(sample_1, sample_2, norm=2, eps=0):
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared).to(torch.float))

def position(H, W, sr_ratio=1, std=1):
    N = H*W
    h = int(H/sr_ratio)
    w = int(W/sr_ratio)
    n = int(h*w)
    yv, xv = torch.meshgrid([torch.arange(H), torch.arange(W)])
    grid0 = torch.stack((xv, yv), 2).view((H, W, 2)).float().transpose(0, 1).cuda()
    grid01 = grid0[:,:,0:1].permute(2,0,1)
    grid02 = grid0[:,:,1:2].permute(2,0,1)
    ymax = F.max_pool2d(grid01, kernel_size=sr_ratio,stride=sr_ratio)
    ymin = -F.max_pool2d(-grid01, kernel_size=sr_ratio,stride=sr_ratio)
    y = ((ymax+ymin) / 2).resize(n,1,1)
    xmax = F.max_pool2d(grid02, kernel_size=sr_ratio,stride=sr_ratio)
    xmin = -F.max_pool2d(-grid02, kernel_size=sr_ratio,stride=sr_ratio)
    x = ((xmax+xmin) / 2).resize(n,1,1)
    grid1 = torch.cat([y,x],2).resize(h,w,2)
    grid0 = grid0.resize(N, 2)
    grid1 = grid1.resize(n, 2)
    dist = pdist(grid0, grid1, norm=2)
    dist1 = dist / (sr_ratio*2)
    dist2 = normpdf(dist1, std)
    dist3 = 10 * dist2.softmax(dim=-1)
    return dist3



