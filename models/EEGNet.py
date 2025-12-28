import torch.nn as nn
import torch


# TODO implement EEGNet model
class EEGNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        chans: int = 2,
        samples: int = 750,
        F1: int = 16,
        D: int = 2,
        F2: int = 32,          # usually F2 = F1 * D
        kernel_length: int = 51,
        separable_kernel: int = 15,
        dropout: float = 0.25,
        elu_alpha: float = 1.0,
    ):
        super().__init__()

        # firstconv
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), stride=(1, 1),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
        )

        # depthwiseConv (spatial filtering)
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(chans, 1), stride=(1, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ELU(alpha=elu_alpha),
            nn.LeakyReLU(negative_slope=0.01),

            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=dropout),
        )

        # separableConv = depthwise temporal + pointwise
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, separable_kernel), stride=(1, 1),
                      padding=(0, separable_kernel // 2), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(F2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ELU(alpha=elu_alpha),
            nn.LeakyReLU(negative_slope=0.01),

            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=dropout),
        )

        # classifier: infer flatten dim by dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            feat = self._forward_features(dummy)
            flatten_dim = feat.view(1, -1).shape[1]

        self.classify = nn.Sequential(
            nn.Linear(flatten_dim, num_classes, bias=True)
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = x.flatten(start_dim=1)
        x = self.classify(x)
        return x

# (Optional) implement DeepConvNet model
class DeepConvNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        chans: int = 2,
        samples: int = 750,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Block 1: temporal conv + spatial conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.Conv2d(25, 25, kernel_size=(chans, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=dropout),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=dropout),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=dropout),
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=dropout),
        )

        # Infer flatten dim
        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            feat = self._forward_features(dummy)
            flatten_dim = feat.view(1, -1).shape[1]

        self.classifier = nn.Linear(flatten_dim, num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x
