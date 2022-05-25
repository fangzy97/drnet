

import torch

def get_model_para_number(model):
    total_number = 0
    for name, para in model.named_parameters():
        if 'cls' in name or 'IOM' in name or 'non' in name or 'side' in name:
            total_number += torch.numel(para)

    return total_number
