from math import floor, ceil
import io
import os

import dill
from sklearn.model_selection import train_test_split
from PIL import Image
from progress.bar import Bar
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms


classes = ['a', 'ba', 'dara', 'ei', 'ga', 'ha', 'ka', 'kuw', 'la', 'ma', 'na', 'nga', 'ou', 'pa', 'sa', 'ta', 'tul', 'wa', 'ya']


DEFAULT_CNN_PARAMS = {
'conv_layer_configs' : [
                        {'filters' : 6,
                        'kernel_size' : (5, 5),
                        'stride' : (1, 1),
                        'pool' : (2, 2),
                        'padding' : 'valid'},
                        {'filters' : 16,
                        'kernel_size' : (5, 5),
                        'stride' : (1, 1),
                        'pool' : (2, 2),
                        'padding' : 'valid'}
                    ],
'fc_layer_configs' : [
                        {'size' : 120},
                        {'size' : 84},
                    ],
'batch_norm' : False,
'dropout' : 0.0,
'activation_fn' : 'ReLU'
}


DEFAULT_TRAIN_PARAMS = {
    'epochs' : 2,
    'batch_size' : 4,
    'optimizer_class' : 'SGD',
    'learning_rate' : 0.001,
    'momentum' : 0.9
}

ALTERNATE_CNN_PARAMS = {
'conv_layer_configs' : [
                        {'filters' : 6,
                        'kernel_size' : (7, 7),
                        'stride' : (1, 1),
                        'pool' : (2, 2),
                        'padding' : 'valid'},
                        {'filters' : 16,
                        'kernel_size' : (5, 5),
                        'stride' : (1, 1),
                        'pool' : (2, 2),
                        'padding' : 'valid'}
                    ],
'fc_layer_configs' : [
                        {'size' : 128},
                        {'size' : 64},
                    ],
'batch_norm' : False,
'dropout' : 0.5,
'activation_fn' : 'ReLU'
}

def same_pad_values(dim, kernel_size, stride=(1, 1)):
    total_h_pad = stride[0]*dim[0]-dim[0]+kernel_size[0]-stride[0]
    smaller_h_pad = floor(total_h_pad / 2)
    larger_h_pad = ceil(total_h_pad / 2)
    total_v_pad = stride[1]*dim[1]-dim[1]+kernel_size[1]-stride[1]
    smaller_v_pad = floor(total_v_pad / 2)
    larger_v_pad = ceil(total_v_pad / 2)
    return (smaller_h_pad, larger_h_pad, smaller_v_pad, larger_v_pad)


def find_new_dim(dim, kernel_size, stride=(1, 1), pad=(0, 0, 0, 0)):
    return (floor((dim[0]+sum(pad[:2])-(kernel_size[0]-1)-1)/stride[0]+1),
          floor((dim[1]+sum(pad[2:])-(kernel_size[1]-1)-1)/stride[1]+1))


class baybayin_net(nn.Module):
    def __init__(self, args):
        super(baybayin_net, self).__init__()

        if args['activation_fn'] == 'Sigmoid':
                activation_fn = nn.Sigmoid()
        elif args['activation_fn'] == 'Tanh':
            activation_fn = nn.Tanh()
        elif args['activation_fn'] == 'ReLU':
            activation_fn = nn.ReLU()
        
        self.conv_layers = []
        dim = (32, 32)
        for i, conv_layer in enumerate(args['conv_layer_configs']):
            if conv_layer['padding'] == 'same':
                self.conv_layers.append(
                    nn.ZeroPad2d(
                        same_pad_values(
                            dim, conv_layer['kernel_size'], conv_layer['stride']
                        )
                    )
                )
            else:
                dim = find_new_dim(dim, conv_layer['kernel_size'], conv_layer['stride'])
      
            self.conv_layers.append(
                nn.Conv2d(
                  in_channels=(3 if i==0 else args['conv_layer_configs'][i-1]['filters']),
                  out_channels=conv_layer['filters'],
                  kernel_size=conv_layer['kernel_size'],
                  stride=conv_layer['stride']
                )
            )

            self.conv_layers.append(activation_fn)

            if args['batch_norm']:
                self.conv_layers.append(nn.BatchNorm2d(conv_layer['filters']))
      
            self.conv_layers.append(nn.MaxPool2d(kernel_size=conv_layer['pool']))
            dim = find_new_dim(dim, conv_layer['pool'], (2, 2))
    
        self.conv_layers = nn.ModuleList(self.conv_layers)
        
        self.fc_layers = []
        num_of_fc_layers = len(args['fc_layer_configs'])
        for i, fc_layer_config in enumerate(args['fc_layer_configs']):
            self.fc_layers.append(
                nn.Linear(
                    dim[0]*dim[1]*args['conv_layer_configs'][-1]['filters'] if i==0 else args['fc_layer_configs'][i-1]['size'],
                    fc_layer_config['size']
                )
            )
            self.fc_layers.append(activation_fn)
            self.fc_layers.append(nn.Dropout(args['dropout']))
        self.fc_layers.append(
            nn.Linear(args['fc_layer_configs'][-1]['size'], 19)
            )
        self.fc_layers = nn.ModuleList(self.fc_layers)
  
    def forward(self, x):
        b = x.shape[0]
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(b, -1)
        for layer in self.fc_layers:
            x = layer(x)
        return x


def pre_process_image(image):
    return transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(
                                lambda x : transforms.Resize(
                                    round(min(x.shape[-2:]) * 32 / max(x.shape[-2:]))
                                )(x)
                            ), # resize so larger dimension is 32
                            transforms.Pad(16, fill=1), # pad dimensions in case current dimensions are less than 32x32
                            transforms.CenterCrop(32), # crop to 32x32
                            transforms.Lambda(
                                lambda x : x.expand(3, -1, -1) if x.shape[0] != 3 else x
                            ), # expand 1 x 32 x 32 image to 3 x 32 x 32
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize between 1 and -1
    ])(image)

def create_transforms():
    return transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(
                                lambda x : transforms.Resize(
                                    round(min(x.shape[-2:]) * 32 / max(x.shape[-2:]))
                                )(x)
                            ), # resize so larger dimension is 32
                            transforms.Pad(16, fill=1), # pad dimensions in case current dimensions are less than 32x32
                            transforms.CenterCrop(32), # crop to 32x32
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize between 1 and -1
    ])


def create_loaders(batch_size):

    tfs = create_transforms()

    print('Creating dataset...', end='')
    if os.path.isfile('baybayin_images.pickle'):
        # load baybayin images dataset from file if it exists
        with open('baybayin_images.pickle', 'rb') as file:
            baybayin_images = dill.load(file)
    else:
        # create baybayin images and save it to a file
        baybayin_images = ImageFolder(root='./Baybayin-Handwritten-Character-Dataset/raw', transform=tfs)
        with open('baybayin_images.pickle', 'wb') as file:
            dill.dump(baybayin_images, file)
    print('Done')

    # split images into train and test sets
    print('Creating train and test sets...', end='')
    if os.path.isfile('baybayin_train.pickle') and os.path.isfile('baybayin_test.pickle'):
        with open('baybayin_train.pickle', 'rb') as file:
            baybayin_train = dill.load(file)
        with open('baybayin_test.pickle', 'rb') as file:
            baybayin_test = dill.load(file)
    else:
        baybayin_train, baybayin_test = train_test_split(baybayin_images, train_size=0.7, stratify=([y for _, y in baybayin_images]), random_state=42)
        with open('baybayin_train.pickle', 'wb') as file:
            dill.dump(baybayin_train, file)
        with open('baybayin_test.pickle', 'wb') as file:
            dill.dump(baybayin_test, file)
    print('Done')

    # create train and test loaders
    baybayin_trainloader = torch.utils.data.DataLoader(baybayin_train, batch_size=batch_size, shuffle=True, num_workers=2)
    baybayin_testloader = torch.utils.data.DataLoader(baybayin_test, batch_size=batch_size, shuffle=True, num_workers=2)

    return baybayin_trainloader, baybayin_testloader


def train_model(cnn_args, train_args, save_path):

    # initialize model
    model = baybayin_net(cnn_args)
    print(model)

    #create train and test loaders
    baybayin_trainloader, baybayin_testloader = create_loaders(train_args['batch_size'])

    # initialize optimizer
    if train_args['optimizer_class'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=train_args['learning_rate'], momentum=train_args['momentum'])
    elif train_args['optimizer_class'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=train_args['learning_rate'], betas=train_args['betas'])

    criterion = nn.CrossEntropyLoss() # initialize loss

    bar = Bar('Training', max = train_args['epochs'] * len(baybayin_trainloader))
    for epoch in range(train_args['epochs']):  # loop over the dataset multiple times

        running_loss = 0
        for i, data in enumerate(baybayin_trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'epoch {epoch:2d}\titer {i:5d}\trunning loss {running_loss/2000:.3f} %.3f')
                running_loss = 0.0

            bar.next()
    torch.save(
        {
        'model' : model.state_dict(),
        'cnn_params' : cnn_args,
        'train_params' : train_args
        },
        save_path
        )
    bar.finish()
    print('Finished Training')


def evaluate_model(cnn_args, train_args, load_path='default.pt'):
    model_and_params = torch.load(load_path)
    if cnn_args is None:
        cnn_args = model_and_params['cnn_params']
    if train_args is None:
        train_args = model_and_params['train_params']
    model = baybayin_net(cnn_args)
    model.load_state_dict(model_and_params['model'])
    print(model)
    _, baybayin_testloader = create_loaders(train_args['batch_size'])
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in baybayin_testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on {len(baybayin_testloader)}-image test set: {100*correct/total:.2f}%')


def classify_uploaded_file(uploaded_file, load_path='default.pt'):
    drawing = uploaded_file.read()
    drawing = pre_process_image(Image.open(io.BytesIO(drawing))).unsqueeze(0)
    model_and_params = torch.load(load_path)
    cnn_args = model_and_params['cnn_params']
    classifier = baybayin_net(cnn_args)
    classifier.load_state_dict(model_and_params['model'])
    print(classifier)
    classifier.eval()
    with torch.no_grad():
        predictions = F.softmax(classifier(drawing), 1)
    probability, class_index = predictions.max(1)
    classification = classes[class_index.item()]
    probability = probability.item()
    print('Prediction:', classification)
    print('Probability:', probability)
    return classification, probability

# train_model(DEFAULT_CNN_PARAMS, DEFAULT_TRAIN_PARAMS, 'default.pt')
# evaluate_model(DEFAULT_CNN_PARAMS, DEFAULT_TRAIN_PARAMS, 'default.pt')