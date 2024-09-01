import torch
import torch.nn as nn
import pandas as pd

class BiLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(BiLSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 because it's bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirection

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # We use the output of the last time step
        return out

def load_data(file_path):
    data = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = lines[1:]  # Skip the header
        for line in lines:
            items = line.strip().split("\t")
            name = items[0]
            features = [float(item) for item in items[1:]]
            data[name] = features
    return data

def save_results(output, data, output_file="lstm.out", result_file="lstm.res"):
    output_df = pd.DataFrame(output.detach().numpy())
    output_df.to_csv(output_file, sep=" ", index=False)
    
    result_df = pd.DataFrame(output_df.idxmax(1))
    result_df.index = list(data.keys())  # Set the index to sample names
    result_df.to_csv(result_file, sep=" ", header=False)

# Initialize model parameters
input_size = 20  # Adjust this according to your actual input size
hidden_size = 64
output_size = 2  # Adjust this according to the number of classes
model = BiLSTMNet(input_size, hidden_size, output_size)

# Load the trained model state
model.load_state_dict(torch.load('lstm_class_model.pth'))
model.eval()

# Load data
data = load_data('DPC.out')
tensor_data = torch.tensor(list(data.values()), dtype=torch.float32).unsqueeze(1)  # Add a dimension for the sequence

# Make predictions
with torch.no_grad():
    output = model(tensor_data)

# Save the results
save_results(output, data)
