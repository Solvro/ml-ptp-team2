import torch
import torch.nn.functional as F


# scaling factor should be used to make this loss in the same
# range as the prediction loss
def tce_loss(y_pred, y_true, err=0.00001, scaling_factor=1 / 12):
    return scaling_factor * torch.mean(((1 + y_true) / 2) * torch.log((1 + y_pred + err) / 2)
                                       + ((1 - y_true) / 2) * torch.log((1 - y_pred + err) / 2))


if __name__ == '__main__':
    input = torch.ones((200, 200, 200))
    pred = torch.ones((200, 200, 200))
    print(tce_loss(pred, input))
    input = torch.ones((200, 200, 200))
    pred = torch.ones((200, 200, 200)) * (-1)
    print(tce_loss(pred, input))
    input = torch.ones((200, 200, 200))
    pred = torch.zeros((200, 200, 200))
    print(tce_loss(pred, input))
    input = torch.ones((200, 200, 200))
    pred = torch.zeros((200, 200, 200)) + 0.5
    print(tce_loss(pred, input))
    print(F.mse_loss(input, pred))
