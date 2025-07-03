from typing import List
import torch
from torch import nn

class DataStore:
    def prepare(self, all_image_token_length, frame_token_length, frame_number, system_token_length,user_instruction_length):
        self.all_image_token_length = int(all_image_token_length)
        self.frame_token_length = int(frame_token_length)
        self.frame_number = int(frame_number)
        self.system_token_length = int(system_token_length)
        self.user_instruction_length=int(user_instruction_length)
        self.image_start_index = int(system_token_length)
        self.image_end_index=int(system_token_length)+int(all_image_token_length)
# 创建一个实例
data = DataStore()