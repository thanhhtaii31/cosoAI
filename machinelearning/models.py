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
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
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
        super().__init__()
        hidden_size_layer = 100
        # Ví dụ: hidden_size_layer = 3
        self.layer_1 = Linear(1, hidden_size_layer) 
        # Bên trong layer_1 có W_1 = [[w1], [w2], [w3]] và b1 = [b1, b2, b3]
        # layer_1 = Linear(1, 3)
        # Tương đương với 
            # h1 = x*w1 + b1
            # h2 = x*w2 + b2
            # h3 = x*w3 + b3

        self.layer_2 = Linear(hidden_size_layer, 1)
        # Bên trong layer_2 có W_2 = [[wa, wb, wc]] và b2 = [b4]
        # layer_2 = Linear(3, 1)
        # Tương đương với
        # y_predicted = h1*wa + h2*wb + h3*wc + b4

    def forward(self, x):
        # Minh hoạ: h = layer_1(x) -> h = ReLU(h) -> y_predicted = layer_2(h)
        # ReLU: Giá trị âm chuyển thành 0, giá trị dương giữ nguyên
        h = self.layer_1(x)
        h = torch.relu(h)
        y_predicted = self.layer_2(h)
        return y_predicted

    def get_loss(self, x, y):
        y_predicted = self.forward(x)
        loss = mse_loss(y_predicted, y)
        # Maybe hàm này trả về theo công thức: loss = mean((y_predicted - y)^2)
        return loss

    # 1. forward   → tính ŷ
    # 2. loss      → đo sai lệch
    # 3. backward  → tính độ thay đổi để sửa
    # 4. optimizer → cập nhập trọng số
    # 5. lặp lại 1000 lần
    def train(self, dataset):
        BATCH_SIZE = 20
        EPOCHS = 1000
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(self.parameters(), 0.005) # Khởi tạo với learning rate = 0.005
        for EPOCH in range(EPOCHS):
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                optimizer.zero_grad()   # xóa gradient cũ
                loss = self.get_loss(x, y)
                loss.backward()         # tính gradient mới 
                optimizer.step()        # cập nhật trọng số tức W = W - 0.005*gradient_mới


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

        self.hidden_2 = Linear(hidden_size, len(self.languages))

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

    rows = []
    for i in range(output_height):
        col_result = []
        for j in range(output_width):
            slicing = input[i: i + weight_dimensions[0], j: j + weight_dimensions[1]]
            result = (slicing * weight).sum()
            col_result.append(result)
        rows.append(torch.stack(col_result))
    Output_Tensor = torch.stack(rows)

    return Output_Tensor


class DigitConvolutionalModel(Module):

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))

        self.hidden_layer = Linear(26 * 26, 10)

    def run(self, x):
        return self(x)

    def forward(self, x):
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)
        x = relu(x)
        x = self.hidden_layer(x)
        return x

    def get_loss(self, x, y):
        x_predicted = self.run(x)
        loss = cross_entropy(x_predicted, y)
        return loss

    def train(self, dataset):
        BATCH_SIZE = 100
        EPOCHS = 20

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

        B, T, C = input.size()

        """YOUR CODE HERE"""
        k = self.k_layer(input)
        q = self.q_layer(input)
        v = self.v_layer(input)

        q_transpose = q.transpose(1,2)

        score = matmul(k,q_transpose)

        score = score/(self.layer_size **0.5)

        score =score.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        weight = softmax(score, dim = -1)

        final = matmul(weight, v)

        return final