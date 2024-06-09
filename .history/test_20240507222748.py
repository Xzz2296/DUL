# import torch

# a = torch.randn(512,85742)
# b = torch.randn(512,512)
# a_mean = torch.mean(a, dim=1, keepdim=True)
# a_mean = a_mean.expand_as(b)

# print(a_mean.shape)

from typing import List
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result =[]
        for i in range(len(nums)):
            if nums[i] > 0:
                break
            if i >0 and nums[i] == nums[i -1]:
                continue
            left =i+1
            right = len(nums) -1
            while left < right :
                sum_value =nums[i] + nums[left] +nums[right]
                if  sum_value> 0:
                    right -=1
                elif sum_value < 0:
                    left += 1
                else:
                    result.append([nums[i],nums[left],nums[right]])
        
        # result =set(result)
        return result