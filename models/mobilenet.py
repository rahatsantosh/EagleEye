import torch.nn as nn
import torch
import math

__all__ = ["mobilenetv2"]


def mobilenetv2(**kwargs):
    import torchvision

    model = torchvision.models.mobilenet_v2(**kwargs)
    return model


if __name__ == "__main__":
    net = mobilenetv2()
    image = torch.randn(2, 3, 224, 224)
    print(net)
    print(net.layer1[1].conv2)
    out = net(image)
    print(out.size())

    # print(distiller.weights_sparsity_summary(net))
