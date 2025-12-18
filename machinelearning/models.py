from sympy.codegen.fnodes import dimension
from torch import no_grad, stack
from torch.ao.nn.quantized import Softmax
from torch.utils.data import DataLoader
from torch.nn import Module

import torch
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, ones, matmul
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        super(PerceptronModel, self).__init__()
        self.w = Parameter(torch.ones(1, dimensions))

    def get_weights(self):
        return self.w

    def run(self, x):
        score = torch.tensordot(x, self.w, dims=([1], [1]))
        return score

    def get_prediction(self, x):
        Score = self.run(x)
        score = float(Score)
        if score >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            "*** YOUR CODE HERE ***"
            flag = False
            while flag == False:
                flag = True
                for batch in dataloader:
                    x = batch['x']
                    y = batch['label']
                    pred = self.get_prediction(x)
                    if pred != y.item():
                        flag = False
                        with torch.no_grad():
                            self.w.data += x * y


class RegressionModel(Module):
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        hidden_size_layer_1 = 400
        hidden_size_layer_2 = 500

        self.layer_1 = Linear(1, hidden_size_layer_1)        #Parameter(torch.rand(1, hidden_size_layer_1))

        self.layer_2 = Linear(hidden_size_layer_1, hidden_size_layer_2)

        self.layer_3 = Linear(hidden_size_layer_2,1)

    def forward(self, x):
        hidden_1 = relu(self.layer_1(x))
        hidden_2 = relu(self.layer_2(hidden_1))
        return self.layer_3(hidden_2)

    def get_loss(self, x, y):
        y_predicted = self.forward(x)
        loss = mse_loss(y_predicted, y)
        return loss

    def train(self, dataset):
        BATCH_SIZE = 20
        EPOCHS = 1000
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(self.parameters(), 0.005)
        for EPOCH in range(EPOCHS):
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()


class DigitClassificationModel(Module):

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        hidden_size = 200

        self.hidden_1 = Linear(input_size, hidden_size)
        # self.w1 = Parameter(torch.rand(input_size, hidden_size))
        # self.b1 = Parameter(torch.zeros(1, hidden_size))

        self.hidden_2 = Linear(hidden_size, output_size)
        # self.w2 = Parameter(torch.rand(hidden_size, output_size))
        # self.b2 = Parameter(torch.zeros(1, output_size))

    def run(self, x):
        hidden_layer_1 = relu(self.hidden_1(x))
        return self.hidden_2(hidden_layer_1)

    def get_loss(self, x, y):
        y_predicted = self.run(x)
        loss = cross_entropy(y_predicted, y)
        return loss

    def train(self, dataset):
        BATCH_SIZE = 30
        EPOCHS = 30
        optimizer = optim.Adam(self.parameters(), 0.005)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        for EPOCH in range(EPOCHS):
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()


class LanguageIDModel(Module):
    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        hidden_size = 300

        self.hidden_1 = Linear(self.num_chars, hidden_size)
        # self.w1 = Parameter(torch.rand(self.num_chars, hidden_size))
        # self.b1 = Parameter(torch.zeros(1, self.hidden_size))
        self.hidden_2 = Linear(hidden_size, len(self.languages))
        # self.w2 = Parameter(torch.rand(hidden_size, len(self.languages)))
        # self.b2 = Parameter(torch.zeros(1, len(self.languages)))

        self.hidden = Linear(hidden_size, hidden_size)

    def run(self, xs):
        batch_size = xs.shape[1]
        hidden_size = self.hidden_1.out_features

        hidden_state = torch.zeros(batch_size, hidden_size)
        for x in xs:
            hidden_state = relu(self.hidden_1(x) + self.hidden(hidden_state))
        scores = self.hidden_2(hidden_state)
        return scores

    def get_loss(self, xs, y):
        scores = self.run(xs)
        y_labels = torch.argmax(y, dim=1)
        loss = cross_entropy(scores, y_labels)
        return loss

    def train(self, dataset):
        EPOCHS = 30
        BATCH_SIZE = 30
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(self.parameters(), 0.005)
        for EPOCH in range(EPOCHS):
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                x_move = movedim(x, 0, 1)
                optimizer.zero_grad()
                loss = self.get_loss(x_move, y)
                loss.backward()
                optimizer.step()


def Convolve(input: tensor, weight: tensor):
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())

    output_height = input_tensor_dimensions[0] - weight_dimensions[0] + 1
    output_width = input_tensor_dimensions[1] - weight_dimensions[1] + 1
    Output_Tensor = torch.zeros(output_height, output_width)

    for i in range(output_height):
        for j in range(output_width):
            result = 0
            for m in range(weight_dimensions[0]):
                for n in range(weight_dimensions[1]):
                    result += input[i + m][j + n] * weight[m][n]
            Output_Tensor[i, j] = result
    return Output_Tensor


class DigitConvolutionalModel(Module):

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))

        self.W = Parameter(torch.rand(26 * 26, 10))

        self.b = Parameter(torch.zeros(1, output_size))

    def run(self, x):
        return self(x)

    def forward(self, x):
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)
        x = relu(x)
        x = matmul(x, self.W) + self.b
        return x

    def get_loss(self, x, y):
        x_predicted = self.run(x)
        y_label = torch.argmax(y, dim=1)
        loss = cross_entropy(x_predicted, y_label)
        return loss

    def train(self, dataset):
        BATCH_SIZE = 5
        EPOCHS = 3

        optimizer = optim.Adam(self.parameters(), lr=0.005)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for EPOCH in range(EPOCHS):
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()


class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)

        # Masking part of attention layer
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

        self.layer_size = layer_size

    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have 
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:
    
        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()

        """YOUR CODE HERE"""
