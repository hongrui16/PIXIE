import torch

class ComputeL2Loss(torch.nn.Module):
    def __init__(self):
        super(ComputeL2Loss, self).__init__()
    
    def forward(self, pred, target, valid_mask=None, confidence = None, reduction='mean'):
        """
        计算L2损失
        参数:
        - pred: 预测值，形状为(B, H, W)。
        - target: 目标值，形状为(B, H, W)。
        - valid_mask: 有效掩码，形状为(B, H) 
        - confidence: 置信度，形状为(B, H)。
        返回:
        - loss: L2损失。
        """
        # print('pred:', pred.shape) # torch.Size([bs, 137, 2])
        # print('target:', target.shape) # torch.Size([bs, 137, 2])
        # print('valid_mask:', valid_mask.shape) # torch.Size([bs, 137])
        # print('confidence:', confidence.shape) # torch.Size([bs, 137])
        # 检查输入的形状是否一致
        assert pred.shape == target.shape, "pred and target must have the same shape"
        bs, h, w = pred.shape
        if valid_mask is not None:
            # 如果valid_mask的形状为(B, H)，将其扩展到(B, H, W)
            if valid_mask.dim() == 2:
                valid_mask = valid_mask.unsqueeze(2).expand_as(pred)
            elif valid_mask.dim() == 3:
                assert valid_mask.shape == pred.shape, "valid_mask must have the same shape as pred and target"
            else:
                raise ValueError("valid_mask must have 2 or 3 dimensions")
            
            # 计算加权的L2损失
            loss = torch.sum((pred - target) ** 2 * valid_mask) / (torch.sum(valid_mask) + 1e-6)
            
        elif confidence is not None:
            # 如果confidence的形状为(B, H)，将其扩展到(B, H, W)
            if confidence.dim() == 2:
                confidence = confidence.unsqueeze(2).expand_as(pred)
            elif confidence.dim() == 3:
                assert confidence.shape == pred.shape, "confidence must have the same shape as pred and target"
            else:
                raise ValueError("confidence must have 2 or 3 dimensions")
            
            # 计算加权的L2损失
            loss = torch.sum((pred - target) ** 2 * confidence)/ (torch.sum(confidence>0) + 1e-6)

        else:
            # 计算普通的L2损失
            loss = torch.mean((pred - target) ** 2)
        
        return loss
