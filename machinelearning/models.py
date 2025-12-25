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

# ===== Q1 =====
class PerceptronModel(Module):
    def __init__(self, dimensions):
        super(PerceptronModel, self).__init__()
        self.w = Parameter(torch.ones(1, dimensions)) # Khởi tạo vector trọng số w với kích thước (1, dimensions)

    def get_weights(self):
        return self.w

    def run(self, x):
        score = torch.tensordot(x, self.w, dims=([1], [1])) # Tính tích vô hướng giữa x và w
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

# ===== Q2 =====
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
        # Hàm này trả về theo công thức: loss = mean((y_predicted - y)^2)
        return loss

    # 1. forward   -> tính ŷ
    # 2. loss      -> đo sai lệch
    # 3. backward  -> tính độ thay đổi để sửa
    # 4. optimizer -> cập nhập trọng số
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

# ===== Q3 =====
class DigitClassificationModel(Module):
    def __init__(self):
        super().__init__()
        input_size = 28 * 28 # Đã được flatten thành vector 784 chiều
        hidden_size = 200
        output_size = 10

        self.hidden_1 = Linear(input_size, hidden_size) # Linear(784, 200)
        # Tương đương với
        # h1 = w1,1 * x1 + w1,2 * x2 + ... + w1,784 * x784 + b1
        # h2 = w2,1 * x1 + w2,2 * x2 + ... + w2,784 * x784 + b2
        # ...
        # h200 = w200,1 * x1 + w200,2 * x2 + ... + w200,784 * x784 + b200
        
        self.hidden_2 = Linear(hidden_size, output_size) # Linear(200, 10)
        # Tương đương với 
        # y1 = w1,1 * h1 + w1,2 * h2 + ... + w1,200 * h200 + c1
        # y2 = w2,1 * h1 + w2,2 * h2 + ... + w2,200 * x200 + c2
        # ...
        # y10 = w10,1 * h1 + w10,2 * h2 + ... + w10,200 * x200 + c200

    def run(self, x):
        hidden_layer_1 = relu(self.hidden_1(x)) 
        return self.hidden_2(hidden_layer_1)

    def get_loss(self, x, y):
        y_predicted = self.run(x) # y_predicted là scores hoặc logits (số thô)
        loss = cross_entropy(y_predicted, y) # loss = −log(p_đúng​)
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

# ===== Q4 =====
class LanguageIDModel(Module):
    def __init__(self):
        # num_chars: số ký tự trong encoding (47), languages: danh sách ngôn ngữ
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        hidden_size = 300

        # Lớp nhận input ký tự -> hidden
        self.hidden_1 = Linear(self.num_chars, hidden_size)

        # Lớp cuối: hidden -> số lớp ngôn ngữ
        self.hidden_2 = Linear(hidden_size, len(self.languages))

        # Một lớp chuyển trạng thái ẩn -> ẩn (giúp tính recurrent)
        self.hidden = Linear(hidden_size, hidden_size)

    def run(self, xs):
        # xs có dạng (sequence_length, batch_size, num_chars)
        batch_size = xs.shape[1]
        hidden_size = self.hidden_1.out_features

        # Khởi tạo trạng thái ẩn cho batch
        hidden_state = torch.zeros(batch_size, hidden_size)
        # Lặp qua từng bước thời gian (từng ký tự trong chuỗi)
        for x in xs:
            # hidden_state = ReLU(hidden_1(x) + hidden(hidden_state))
            hidden_state = relu(self.hidden_1(x) + self.hidden(hidden_state))
        # Sau khi xử lý toàn bộ chuỗi, ánh xạ về scores cho từng ngôn ngữ
        scores = self.hidden_2(hidden_state)
        return scores

    def get_loss(self, xs, y):
        # y có thể là one-hot; torch.argmax chuyển về chỉ số lớp
        scores = self.run(xs)
        y_labels = torch.argmax(y, dim=1) # Tìm giá trị lớn nhất trong one-hot vector
        loss = cross_entropy(scores, y_labels) # −log(p_đúng​)
        return loss

    def train(self, dataset):
        # Huấn luyện: đưa batch về dạng (seq_len, batch, chars) bằng movedim
        EPOCHS = 30
        BATCH_SIZE = 30
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(self.parameters(), 0.005)
        for EPOCH in range(EPOCHS):
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                # movedim đổi kích thước để bước thời gian ở dim0
                x_move = movedim(x, 0, 1)
                optimizer.zero_grad()
                loss = self.get_loss(x_move, y)
                loss.backward()
                optimizer.step()

# ===== Q5 =====
# Convolve: hàm convolution 2D thủ công (không dùng torch.nn.Conv2d)
def Convolve(input: tensor, weight: tensor):
    # input: ảnh 2D (height, width), weight: kernel 2D
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())

    # Kích thước output = input - kernel + 1 (valid convolution)
    output_height = input_tensor_dimensions[0] - weight_dimensions[0] + 1
    output_width = input_tensor_dimensions[1] - weight_dimensions[1] + 1

    rows = []
    for i in range(output_height):
        col_result = []
        for j in range(output_width):
            # Lấy miếng con của ảnh tương ứng kernel
            slicing = input[i: i + weight_dimensions[0], j: j + weight_dimensions[1]]
            # Nhân element-wise rồi cộng tất cả (dot product)
            result = (slicing * weight).sum()
            col_result.append(result)
        rows.append(torch.stack(col_result))
    Output_Tensor = torch.stack(rows)

    return Output_Tensor

# ===== Q6 =====
class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()

        # Lớp k layer, đại diện cho thông số key, là nhãn mô tả
        self.k_layer = Linear(layer_size, layer_size)
        # Lớp k layer, đại diện cho thông số query, đi tìm các k ph hợp với từ
        self.q_layer = Linear(layer_size, layer_size)
        # lớp v layer, đại diện cho thông số value, là nội dung của từ đấy
        # Nếu q và k khớp nhau, mô hình sẽ lấy thông tin từ v, còn không khớp thì bỏ qua
        self.v_layer = Linear(layer_size, layer_size)

        # Masking part of attention layer
        # Tạo 1 buffer để lưu trạng thái của mô hình mà không học. Không tính đạo hàm
        # Tạo 1 ma trận tam giác dưới từ hàm torch.tril để đảm bảo hàng 1 chỉ nhìn thấy nó
        # Hàng 2 chỉ nhìn thấy từ thứ 1, và 2, hàng 3 thì có thể thay cả 3 từ
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))
        # Lấy kích thước của layer
        self.layer_size = layer_size

    def forward(self, input):
        # Trích xuất cấu trúc của dữ liệu đầu vào
        # B là batch size: kích thước lô
        # T là time steps: Chiều dài của chuỗi
        # C là chanel: Số kênh đặc trưng
        B, T, C = input.size()

        # tạo ma trận k đại diện cho vai trò nhãn
        k = self.k_layer(input)
        # Tạo ma trận q đại diện cho vai trò query
        q = self.q_layer(input)
        # Tạo ma trận v đại diện cho vai trò value
        v = self.v_layer(input)

        # thục hiện phép chuyển vị bằng hàm transpose
        q_transpose = q.transpose(1,2)

        # Giá trị score theo công thức = tích vô hướng ma trận chuyển vị q và k
        score = matmul(k,q_transpose)

        # Scaling score về giá trị thấp hơn nhằm đảm bảo số không quá to khi tính toán
        score = score/(self.layer_size **0.5)

        # Sử dụng masked_fill, ma trân self.mask ( ma trận tam giác dưới) sẽ che các ô tương lai
        # Những ô trong mask có giá trị bằng 0 sẽ bị gán bằng - vô cùng
        # Tức là khi tính xác suất, mô hình tại thời điểm t sẽ không thể thấy mô hình ở thời điểm t+1
        score =score.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        # Hàm Softmax được áp dụng trên chiều cuối cùng để chuyển đổi các điểm số thành xác suất
        #
        weight = softmax(score, dim = -1)

        # lấy trọng số nhân với ma trận Value
        # Mỗi vector tại vị trí $t$ không còn đơn thuần là vector của chính nó,
        # mà là một vector ngữ cảnh đã được tổng hợp thông tin từ những từ quan trọng nhất đứng trước nó
        final = matmul(weight, v)

        return final