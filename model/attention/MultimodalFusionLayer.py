import torch.nn

from model.TRAR.cls_layer import cls_layer_both


class MultimodalFusionLayer(torch.nn.Module):
    def __init__(self, opt):
        super(MultimodalFusionLayer, self).__init__()
        self.cls_layer = cls_layer_both(opt["hidden_size"], opt["output_size"])

    def forward(self, text_incongruity, image_incongruity, text_memory, image_memory):
        text_incongruity = torch.mean(text_incongruity, dim=1)
        image_incongruity = torch.mean(image_incongruity, dim=1)
        text_memory = torch.mean(text_memory, dim=1)
        image_memory = torch.mean(image_memory, dim=1)

        out1 = self.cls_layer(text_incongruity, image_incongruity)
        out2 = self.cls_layer(text_memory, image_memory)
        out = self.cls_layer(out1, out2)
        return out
