import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        # YOUR CODE HERE (done)
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        in_channels=3
        out_channels=16
        kernel_size=3
        
        # note: input = 224x224 pixels, RGB

        self.model_ = nn.Sequential(
            # nn.Conv2d(3,10, kernel_size, padding=1),
            # nn.Conv2d(3,10, kernel_size, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(3,10, kernel_size, padding=1),
            # nn.Conv2d(3,10, kernel_size, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Flatten(),

            # # TODO: continue




            # another architecture (VGG11)
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5), # nn.Dropout2d(0.5),

            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(in_features=4096, out_features=num_classes)
        )
        
        self.model = nn.Sequential(
            
            # =================  Backbone ==================#

            # 1st conv+ batchnorm + maxpool + relu+ dropout
            nn.Conv2d(in_channels,out_channels, kernel_size, padding=1), #nn.Conv2d(3, 16, 3, padding=1), # 
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2,2), # -> out_channels x 112 x 112
            nn.ReLU(),
            # nn.Dropout(dropout),
         
            # 2nd conv+ batchnorm + maxpool + relu+ dropout
            nn.Conv2d(out_channels, out_channels*2, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels*2),
            nn.MaxPool2d(2,2),  # -> (out_channels x 2) x 56 x 56 
            nn.ReLU(),
            # nn.Dropout(dropout),
            
            # # 3rd conv+ batchnorm + maxpool + relu+ dropout
            # nn.Conv2d(out_channels*2, out_channels*4, kernel_size, padding=1),
            # nn.BatchNorm2d(out_channels*4),
            # nn.MaxPool2d(2,2), # -> (out_channels x 4) x 28 x 28
            # nn.ReLU(),
            # # nn.Dropout(dropout),

            # 3rd conv+ batchnorm + maxpool + relu+ dropout
            nn.Conv2d(out_channels*2, out_channels*4, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels*4, out_channels*4, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels*4),
            nn.MaxPool2d(2,2), # -> (out_channels x 4) x 28 x 28
            nn.ReLU(),
            # nn.Dropout(dropout),

            # 4th conv+ batchnorm + maxpool + relu+ dropout
            nn.Conv2d(out_channels*4, out_channels*8, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels*8),
            nn.MaxPool2d(2,2), # -> (out_channels x 8) x 14 x 14
            nn.ReLU(),
            # nn.Dropout(dropout),

            # # 5th conv+ batchnorm + maxpool + relu+ dropout
            # nn.Conv2d(out_channels*8, out_channels*16, kernel_size, padding=1),
            # nn.BatchNorm2d(out_channels*16),
            # nn.MaxPool2d(2,2), # -> (out_channels x 16) x 7 x 7
            # nn.ReLU(),
            # # nn.Dropout(dropout),
            

            # Flatten features maps
            nn.Flatten(), # -> (out_channels x 16) x 7 x 7 = 12,544  # ->  1 x  (out_channels x 8) x 14 x 14 = 25,088 # ->  1 x  (out_channels x 4) x 28 x 28 = 50,176
            # nn.Dropout(dropout),
            
            
             # =================  Head ==================#
#           # Old head  
#             nn.Linear(50176, 20350),
#             nn.BatchNorm1d(20350),
#             nn.ReLU(),
#             nn.Dropout(dropout),
            
#             nn.Linear(20350, 10000),
#             nn.BatchNorm1d(10000),
#             nn.ReLU(),
#             nn.Dropout(dropout),
            
#             nn.Linear(10000, 3400),
#             nn.BatchNorm1d(3400),
#             nn.ReLU(),
#             nn.Dropout(dropout),
           

            nn.Linear(25088, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(dropout),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: (done)
        # process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)  #return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    # https://stackoverflow.com/questions/71261347/runtimeerror-dataloader-worker-exited-unexpectedly
    return get_data_loaders(batch_size=2, num_workers=0) #return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
