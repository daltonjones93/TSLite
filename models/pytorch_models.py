import itertools
import json
import math

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error  # or another suitable metric
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class denseFFN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        n_hidden_units,
        n_layers,
        dropout_param=0.05,
        **kwargs,
    ):
        super().__init__()
        self.l1 = nn.Linear(input_size, n_hidden_units)
        self.l2 = nn.Linear(n_hidden_units, output_size)

        self.middle_layers = nn.ModuleList()
        for i in range(n_layers):
            self.middle_layers.append(nn.Linear(n_hidden_units, n_hidden_units))

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_param)

    def forward(self, x):
        x = self.activation(self.l1(x))

        for lin_map in self.middle_layers:
            x = self.activation(lin_map(x))
        x = self.dropout(x)
        return self.l2(x)


class SimpleAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size, num_heads=8, causal=False):
        super().__init__()

        # Make sure hidden_size is divisible by 2

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        # Positional embedding (learnable)
        self.positional_embedding = nn.Embedding(2000, self.hidden_size)
        self.is_causal = causal

    def forward(self, hidden_states: torch.Tensor):

        bsz, q_len, _ = hidden_states.size()
        positions = torch.arange(q_len).unsqueeze(0).expand(bsz, q_len)

        # Add positional embeddings to the input embeddings
        hidden_states = hidden_states + self.positional_embedding(positions)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if self.is_causal:
            mask = torch.triu(torch.ones(q_len, q_len), diagonal=1).bool()
            # Expand the mask to match the shape of the attention weights
            # The mask is of shape (seq_length, seq_length), we need (batch_size, num_heads, seq_length, seq_length)
            mask = mask.unsqueeze(0).unsqueeze(
                0
            )  # Shape: (1, 1, seq_length, seq_length)
            mask = mask.expand(
                bsz, self.num_heads, -1, -1
            )  # Shape: (batch_size, num_heads, seq_length, seq_length)

            # Apply the mask to the attention weights
            # Set the future positions (masked) to a very large negative value to make them zero after softmax
            attn_weights.masked_fill_(mask, float("-inf"))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class pytorchModel:
    # TODO add a variable for loss type, clip norm
    def __init__(self, model_type, verbose=True, **kwargs):
        if model_type == "dense":
            self.model = denseFFN(**kwargs)
        elif model_type == "transformer":
            self.model = TransformerModel(**kwargs)
        elif model_type == "lstm":
            self.model = LSTMModel(**kwargs)
        else:
            raise ValueError("Model Type not recognized")

        self.model_type = model_type

        self.batch_size = kwargs.get("batch_size", None)
        self.lr = kwargs.get("lr", None)
        self.epochs = kwargs.get("epochs", None)
        if not self.epochs or not self.lr or not self.batch_size:
            raise ValueError("Check input, kwargs not set properly")
        self.verbose = verbose

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        if isinstance(X_train, np.ndarray):
            X_train, y_train = (
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).float(),
            )
        if isinstance(X_test, np.ndarray):
            X_test, y_test = (
                torch.from_numpy(X_test).float(),
                torch.from_numpy(y_test).float(),
            )
        criterion = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()

        for epoch in range(self.epochs):
            tk0 = tqdm(dataloader, total=int(len(dataloader)))
            counter = 0
            running_loss = 0
            if self.model_type in ["transformer5", "lstm5"]:  # recurrent training
                for i, (X_batch, y_batch) in enumerate(tk0):
                    for j in range(1, X_batch.shape[1]):

                        X_batch_tmp = X_batch[:, -j:, :]
                        optimizer.zero_grad()
                        output = self.model(X_batch_tmp)
                        loss = criterion(output, y_batch)  # .unsqueeze(1))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        optimizer.step()
                    running_loss += loss.item()
                    counter += 1
                    tk0.set_postfix(loss=(running_loss / (counter)))
            else:
                for i, (X_batch, y_batch) in enumerate(tk0):

                    optimizer.zero_grad()
                    output = self.model(X_batch)
                    loss = criterion(output, y_batch)  # .unsqueeze(1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    running_loss += loss.item()
                    counter += 1
                    tk0.set_postfix(loss=(running_loss / (counter)))

            if isinstance(X_test, torch.Tensor):
                if epoch % 10 == 0:
                    y_pred_test = self.model(X_test)
                    mse_test = mean_squared_error(
                        y_pred_test.detach().numpy().flatten(),
                        y_test.detach().numpy().flatten(),
                    )
                    print(
                        "Epoch: "
                        + str(epoch)
                        + ", Loss: "
                        + str(loss.item())
                        + ", Test Loss: "
                        + str(mse_test)
                    )
            else:
                print("Epoch: " + str(epoch) + ", Loss: " + str(loss.item()))

    def predict(self, X_test):
        if isinstance(X_test, np.ndarray):
            # Convert NumPy array to PyTorch tensor
            X_test = torch.from_numpy(X_test).float()

        self.model.eval()

        with torch.inference_mode():
            output = self.model(X_test)
        output = output.detach().numpy()
        if output.shape[1] == 1:
            return output.squeeze()
        else:
            return output


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        n_layers,
        dropout_param=0.05,
        max_len=2000,
        **kwargs,
    ):
        super().__init__()

        self.ffn0 = denseFFN(input_size, hidden_size, hidden_size, 1)
        self.attention_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norms_attn = nn.ModuleList()
        self.norms_ffn = nn.ModuleList()
        for i in range(n_layers):
            self.attention_layers.append(SimpleAttention(hidden_size))
            self.ffn_layers.append(
                denseFFN(hidden_size, hidden_size, hidden_size * 2, 1)
            )
            self.norms_attn.append(nn.LayerNorm(hidden_size))
            self.norms_ffn.append(nn.LayerNorm(hidden_size))
        self.dropout = nn.Dropout(dropout_param)
        self.last_linear = nn.Linear(hidden_size, output_size)
        self.max_len = max_len

    def forward(self, x):
        if x.shape[1] > self.max_len:
            raise ValueError(
                "x is too long for the model, try to shorten the sequence of inputs"
            )
        x = self.ffn0(x)
        for i in range(len(self.attention_layers)):
            attn = self.attention_layers[i]
            ffn = self.ffn_layers[i]
            norm_attn = self.norms_attn[i]
            norm_ffn = self.norms_ffn[i]
            x = norm_attn(x + self.dropout(attn(x)))
            x = norm_ffn(x + self.dropout(ffn(x)))
        x = x[:, -1, :]
        x = self.last_linear(x)
        return x


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        n_layers,
        dropout_param=0.05,
        **kwargs,
    ):
        super().__init__()

        self.ffn0 = denseFFN(input_size, hidden_size, hidden_size, 1)
        self.lstm_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norms_lstm = nn.ModuleList()
        self.norms_ffn = nn.ModuleList()
        for i in range(n_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=hidden_size, hidden_size=hidden_size, batch_first=True
                )
            )
            self.ffn_layers.append(
                denseFFN(hidden_size, hidden_size, hidden_size * 2, 1)
            )
            self.norms_lstm.append(nn.LayerNorm(hidden_size))
            self.norms_ffn.append(nn.LayerNorm(hidden_size))
        self.dropout = nn.Dropout(dropout_param)
        self.last_linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = self.ffn0(x)
        for i in range(len(self.lstm_layers)):
            lstm = self.lstm_layers[i]
            ffn = self.ffn_layers[i]
            norm_lstm = self.norms_lstm[i]
            norm_ffn = self.norms_ffn[i]
            tmp, _ = lstm(x)
            x = norm_lstm(x + self.dropout(tmp))
            x = norm_ffn(x + self.dropout(ffn(x)))
        x = x[:, -1, :]
        x = self.last_linear(x)
        return x


# Function to perform grid search and save best params to JSON
def grid_search(
    xTrain,
    yTrain,
    xVal,
    yVal,
    param_grid,
    model_type="transformer",
    output_json="best_params.json",
):
    best_params = None
    best_val_score = float("-inf")  # We want to maximize the score

    # Generate all combinations of hyperparameters
    all_combinations = list(itertools.product(*param_grid.values()))

    for combination in all_combinations:
        params = dict(zip(param_grid.keys(), combination))
        print(f"Evaluating with parameters: {params}")

        # Create the model with the current set of parameters
        model = pytorchModel(model_type=model_type, **params)
        model.fit(xTrain, yTrain)

        # Evaluate the model on validation set
        model.model.eval()
        with torch.no_grad():
            ypred = model.predict(xVal.numpy())
            val_score = mean_squared_error(
                yVal.numpy(), ypred
            )  # Use your chosen metric here

        print(f"Validation score for this run: {val_score}")

        # Update best model if current model is better
        if val_score > best_val_score:
            best_val_score = val_score
            best_params = params

    print(f"Best parameters found: {best_params} with score {best_val_score}")
    output_json = model_type + "_" + output_json
    # Write the best parameters to a JSON file
    with open(output_json, "w") as json_file:
        json.dump(best_params, json_file, indent=4)
    print(f"Best parameters written to {output_json}")

    return best_params, best_val_score


if __name__ == "__main__":
    # Assuming train_data, train_labels, val_data, val_labels are already prepared as tensors
    # train_data = ...
    # train_labels = ...
    # val_data = ...
    # val_labels = ...
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from dataset_timeseries import GridmaticTimeseries

    t = GridmaticTimeseries(
        "data.csv", target_col="CAISO_system_load", time_index="interval_start_time"
    )
    t.create_weekday_feature()
    t.create_month_feature(convert_to_cyclic=True)
    t.create_hour_feature(convert_to_cyclic=True)
    # t.create_seasonal_feature()
    # t.create_fft_feature()
    t.fill_missing_values()
    t.scale_features()
    n_steps_predict = 24
    past_states = 128
    xTrain, xVal, xTest, yTrain, yVal, yTest = t.return_training_data_recurrent(
        past_states, n_steps_predict
    )
    input_shape = xTrain.shape[-1]

    param_grid = {
        "input_size": [12],  # Fixed input size
        "output_size": [24],  # Fixed output size (for example, binary classification)
        "hidden_size": [4, 8],  # Try hidden sizes of 32 and 64
        "n_layers": [1, 2],  # Try 1 or 2 LSTM layers
        "lr": [0.001, 0.0001],  # Try learning rates of 0.001 and 0.0001
        "epochs": [1],  # Train for 10 or 20 epochs
        "batch_size": [4],  # Try batch sizes of 32 and 64
    }

    best_params, best_score = grid_search(
        xTrain, yTrain, xVal, yVal, param_grid, model_type="transformer"
    )
