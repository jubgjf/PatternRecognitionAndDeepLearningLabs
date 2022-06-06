import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRNN(nn.Module):
    def __init__(self, weight, input_size, hidden_size, output_size, device):
        super(TextRNN, self).__init__()

        self.hidden_size = hidden_size

        self.weight = weight
        self.device = device

        self.x2h = nn.Linear(input_size, hidden_size).to(device)
        self.h2h = nn.Linear(hidden_size, hidden_size).to(device)
        self.h2y = nn.Linear(hidden_size, output_size).to(device)

        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

        nn.init.normal_(self.x2h.weight * 0.1)
        nn.init.normal_(self.h2h.weight * 0.1)
        nn.init.normal_(self.h2y.weight * 0.1)

    def forward(self, inputs):
        hidden = torch.zeros((inputs.shape[0], self.hidden_size)).to(self.device)

        inputs_embedding = F.embedding(inputs, self.weight)
        inputs_embedding = inputs_embedding.permute(1, 0, 2)

        for x in inputs_embedding:
            hidden = self.tanh(self.x2h(x) + self.h2h(hidden[:x.shape[0]]))

        output = self.softmax(self.h2y(hidden))

        return output


class TextGRU(nn.Module):
    def __init__(self, weight, input_size, hidden_size, output_size, device):
        super(TextGRU, self).__init__()

        self.hidden_size = hidden_size

        self.weight = weight
        self.device = device

        self.x2h = nn.Linear(input_size, 3 * hidden_size).to(device)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size).to(device)
        self.h2y = nn.Linear(hidden_size, output_size).to(device)

        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

        nn.init.orthogonal_(self.x2h.weight)
        nn.init.orthogonal_(self.h2h.weight)
        nn.init.orthogonal_(self.h2y.weight)

    def forward(self, inputs):
        hidden = torch.zeros((inputs.shape[0], self.hidden_size)).to(self.device)

        inputs_embedding = F.embedding(inputs, self.weight)
        inputs_embedding = inputs_embedding.permute(1, 0, 2)

        for x in inputs_embedding:
            x_t = self.x2h(x)
            h_t = self.h2h(hidden)
            x_reset, x_upd, x_new = x_t.chunk(3, 1)
            h_reset, h_upd, h_new = h_t.chunk(3, 1)

            reset_gate = torch.sigmoid(x_reset + h_reset)
            update_gate = torch.sigmoid(x_upd + h_upd)
            new_gate = torch.tanh(x_new + (reset_gate * h_new))

            hy = update_gate * hidden + (1 - update_gate) * new_gate

        output = self.softmax(self.h2y(hy))

        return output


class TextLSTM(nn.Module):
    def __init__(self, weight, input_size, hidden_size, output_size, device):
        super(TextLSTM, self).__init__()

        self.hidden_size = hidden_size

        self.weight = weight
        self.device = device

        self.x2h = nn.Linear(input_size, 4 * hidden_size).to(self.device)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size).to(self.device)
        self.h2y = nn.Linear(hidden_size, output_size).to(device)

        nn.init.orthogonal_(self.x2h.weight)
        nn.init.orthogonal_(self.h2h.weight)
        nn.init.orthogonal_(self.h2y.weight)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        hx = torch.zeros((inputs.shape[0], self.hidden_size)).to(self.device)
        cx = torch.zeros((inputs.shape[0], self.hidden_size)).to(self.device)

        inputs_embedding = F.embedding(inputs, self.weight)
        inputs_embedding = inputs_embedding.permute(1, 0, 2)

        for x in inputs_embedding:
            x = x.view(-1, x.size(1))

            gates = self.x2h(x) + self.h2h(hx)

            gates = gates.squeeze()

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

            hy = torch.mul(outgate, torch.tanh(cy))

        output = self.softmax(self.h2y(hy))

        return output


class TextBiLSTM(nn.Module):
    def __init__(self, weight, input_size, hidden_size, output_size, device):
        super(TextBiLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.weight = weight
        self.device = device

        self.x2h_f = nn.Linear(input_size, 4 * hidden_size).to(self.device)
        self.h2h_f = nn.Linear(hidden_size, 4 * hidden_size).to(self.device)

        self.x2h_b = nn.Linear(input_size, 4 * hidden_size).to(self.device)
        self.h2h_b = nn.Linear(hidden_size, 4 * hidden_size).to(self.device)

        self.h2y = nn.Linear(hidden_size, output_size).to(device)

        nn.init.orthogonal_(self.x2h_f.weight)
        nn.init.orthogonal_(self.h2h_f.weight)
        nn.init.orthogonal_(self.x2h_b.weight)
        nn.init.orthogonal_(self.h2h_b.weight)
        nn.init.orthogonal_(self.h2y.weight)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        hx_f = torch.zeros((inputs.shape[0], self.hidden_size)).to(self.device)
        cx_f = torch.zeros((inputs.shape[0], self.hidden_size)).to(self.device)
        hx_b = torch.zeros((inputs.shape[0], self.hidden_size)).to(self.device)
        cx_b = torch.zeros((inputs.shape[0], self.hidden_size)).to(self.device)

        inputs_embedding = F.embedding(inputs, self.weight)
        inputs_embedding = inputs_embedding.permute(1, 0, 2)

        for x in inputs_embedding:
            x = x.view(-1, x.size(1))
            gates = self.x2h_f(x) + self.h2h_f(hx_f)
            gates = gates.squeeze()
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)
            cy = torch.mul(cx_f, forgetgate) + torch.mul(ingate, cellgate)
            hy_f = torch.mul(outgate, torch.tanh(cy))
        for x in list(reversed(inputs_embedding)):
            x = x.view(-1, x.size(1))
            gates = self.x2h_b(x) + self.h2h_b(hx_b)
            gates = gates.squeeze()
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)
            cy = torch.mul(cx_b, forgetgate) + torch.mul(ingate, cellgate)
            hy_b = torch.mul(outgate, torch.tanh(cy))

        hy = hy_f + hy_b
        output = self.softmax(self.h2y(hy))

        return output


class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(WeatherLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.device = device

        # ingate
        self.x2h_i = nn.Linear(input_size, self.hidden_size)
        self.h2h_i = nn.Linear(self.hidden_size, self.hidden_size)
        # forgetgate
        self.x2h_f = nn.Linear(input_size, self.hidden_size)
        self.h2h_f = nn.Linear(self.hidden_size, self.hidden_size)
        # cellgate
        self.x2h_c = nn.Linear(input_size, self.hidden_size)
        self.h2h_c = nn.Linear(self.hidden_size, self.hidden_size)
        # outgate
        self.x2h_o = nn.Linear(input_size, self.hidden_size)
        self.h2h_o = nn.Linear(self.hidden_size, self.hidden_size)

        self.h2y = nn.Linear(self.hidden_size, 1)

    def forward(self, inputs):
        inputs0 = inputs[0].permute(1, 0, 2)
        inputs1 = inputs[1].permute(1, 0)

        hx = torch.zeros((inputs0.shape[1], self.hidden_size), device=self.device)
        cx = torch.zeros((inputs0.shape[1], self.hidden_size), device=self.device)

        for x in inputs0:
            ingate = torch.sigmoid(self.x2h_i(x) + self.h2h_i(hx))
            forgetgate = torch.sigmoid(self.x2h_f(x) + self.h2h_f(hx))
            outgate = torch.sigmoid(self.x2h_o(x) + self.h2h_o(hx))
            cellgate = torch.tanh(self.x2h_c(x) + self.h2h_c(hx))

            cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
            hx = torch.mul(outgate, torch.tanh(cy))

        hy = self.h2y(hx)
        for i in range(self.output_size - 1):
            x = inputs1[i].view(-1, 1)
            x = nn.Linear(1, self.input_size, device=self.device)(x)

            ingate = torch.sigmoid(self.x2h_i(x) + self.h2h_i(hx))
            forgetgate = torch.sigmoid(self.x2h_f(x) + self.h2h_f(hx))
            outgate = torch.sigmoid(self.x2h_o(x) + self.h2h_o(hx))
            cellgate = torch.tanh(self.x2h_c(x) + self.h2h_c(hx))

            cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
            hx = torch.mul(outgate, torch.tanh(cy))

            hy = torch.cat((hy, self.h2y(hx)), 1)

        return hy
