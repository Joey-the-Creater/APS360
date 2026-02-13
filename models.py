class SPPLayer(nn.Module):
    def __init__(self, pyramid_levels=(1, 2, 4)):
        super().__init__()
        self.pyramid_levels = pyramid_levels

    def forward(self, x):
        bs, c, h, w = x.size()
        outputs = []
        for level in self.pyramid_levels:
            pooled = F.adaptive_max_pool2d(x, (level, level)).contiguous() 
            outputs.append(pooled.view(bs, -1))
        return torch.cat(outputs, dim=1)  


class VGG19_SPP_MLP(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        base = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = base.features

        # Freeze backbone
        for p in self.features.parameters():
            p.requires_grad = False

        self.spp = SPPLayer((1, 2, 4)) 
        in_dim = 512 * 21

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.spp(x)
        return self.classifier(x)


class ResNet18_SPP_MLP(nn.Module):
  
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-2])

        # Freeze backbone
        for p in self.features.parameters():
            p.requires_grad = False

        self.spp = SPPLayer((1, 2, 4))
        in_dim = 512 * 21

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.spp(x)
        return self.classifier(x)