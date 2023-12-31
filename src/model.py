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
        md_kernel_size=5
        lg_kernel_size=7
        
        # note: input = 224x224 pixels, RGB

        self.model_1 = nn.Sequential(

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
        
        self.model_2 = nn.Sequential(
            
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
            
            # 3rd conv+ batchnorm + maxpool + relu+ dropout
            nn.Conv2d(out_channels*2, out_channels*4, md_kernel_size, padding=2),
            nn.BatchNorm2d(out_channels*4),
            nn.MaxPool2d(2,2), # -> (out_channels x 4) x 28 x 28
            nn.ReLU(),
            # nn.Dropout(dropout),

            # 4th conv+ batchnorm + maxpool + relu+ dropout
            nn.Conv2d(out_channels*4, out_channels*8, md_kernel_size, padding=2),
            nn.BatchNorm2d(out_channels*8),
            nn.MaxPool2d(2,2), # -> (out_channels x 8) x 14 x 14
            nn.ReLU(),
            # nn.Dropout(dropout),
            

            # Flatten features maps
            nn.Flatten(), # ->  1 x  (out_channels x 8) x 14 x 14 = 25,088 # ->  1 x  (out_channels x 4) x 28 x 28 = 50,176
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
            #nn.Dropout(dropout),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Dropout(dropout),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, num_classes),
        )
        
        # input image: 224x224
        self.model_3 = nn.Sequential(
            nn.Conv2d(3, 32, (3,5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, (3,5), padding=(1,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2), # output size: 112x112
            
            nn.Conv2d(64, 128, (3,7), padding=(1,3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2), # output size: 56x56
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2), # output size: 28x28
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2), # output size: 14x14
            
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            nn.Flatten(),
            
            nn.Linear(1024 * 14 * 14, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, num_classes),
            
        )
        
        # input image: 224x224
        self.model = nn.Sequential(
        
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output size: 112x112
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output size: 56x56
            nn.BatchNorm2d(32),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output size: 28x28
            nn.BatchNorm2d(64),
            
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2), #nn.ReLU(), # 
            nn.MaxPool2d(2, 2), # output size: 14x14
            nn.BatchNorm2d(128),
            
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2), # nn.ReLU(), # 
            nn.MaxPool2d(2, 2), # output size: 7x7
            nn.BatchNorm2d(256),
            
            nn.Flatten(),
            
            nn.Dropout(p=dropout),
            
            nn.Linear(7 * 7 * 256, 500),
            nn.LeakyReLU(negative_slope=0.2), # nn.ReLU(), # 
            nn.BatchNorm1d(500),
            nn.Dropout(p=dropout),
            
            nn.Linear(500, 256),
            nn.LeakyReLU(negative_slope=0.2), # nn.ReLU(), # 
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout),
            
            nn.Linear(256, num_classes)
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
