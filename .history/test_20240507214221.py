# import torch

# a = torch.randn(512,85742)
# b = torch.randn(512,512)
# a_mean = torch.mean(a, dim=1, keepdim=True)
# a_mean = a_mean.expand_as(b)

# print(a_mean.shape)

from typing import List
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        idx_nz =0
        for i in range(len(nums)):
            while nums[idx_nz] == 0 and idx_nz < len(nums):
                idx_nz += 1
            if idx_nz <len(nums)-1:
                nums[i] =nums[idx_nz]
                idx_nz +=1
            else:
                nums[i] = 0
        return nums            
            