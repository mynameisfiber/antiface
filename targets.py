from lightcnn.light_cnn import LightCNN_29Layers_v2
import torch


class LightCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_classes = 80013
        self.model = LightCNN_29Layers_v2(num_classes=self.n_classes)
        self.model.eval()
        checkpoint = torch.load("./models/lightcnn/"
                                "LightCNN_29Layers_V2_checkpoint.pth.tar")
        self.model.load_state_dict({k.replace('module.', ''): v
                                    for k, v in checkpoint['state_dict'].items()})

    def forward(self, X):
        _, features = self.model(X)
        return features


if __name__ == "__main__":
    import align
    print("loading data")
    images = align.align_image_dir("./images/zuck/original")
    data = torch.autograd.Variable(align.images_to_tensor(images))

    print("loading model")
    model = LightCNN()

    print("Eval")
    print(model(data))
