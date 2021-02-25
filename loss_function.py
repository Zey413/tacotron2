from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \                #mel谱loss由两个构成，第一个是Decoder得到的，第二个是再经过Postnet（一些CNN）后得到的更精细的mel谱
            nn.MSELoss()(mel_out_postnet, mel_target)                   #两个mel谱都计算损失后加和
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)       #Stop Token 是否应该停下来 二分类
        return mel_loss + gate_loss
