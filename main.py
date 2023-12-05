## Main file for running the dimensionality reduction/training the model
from mlp import MLP
import os
import torch
import torch.nn as nn
from pathlib import Path
import shutil
from tqdm import tqdm
import dill as pickle

losses = {'l1': nn.L1Loss(), 'l2': nn.MSELoss()}

def clear_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def train(mlp, train_dataloader, val_dataloader, test_dataloader, train_epochs=1000,
          val_interval = 10, lr=0.001, loss='l1', device='cpu', test_model = 'best',
          save_path="./outputs/test", continue_train = False):

    Path(save_path).mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    lossfcn = losses[loss]

    if not continue_train:
        clear_directory(save_path)
        train_history = []
        val_history = []
        best_val_loss = float('inf')
    else:
        # Load optimizer state (either best or latest)
        optim_filename = f'{continue_train}_optim.pth'
        # NOTE: Network load dir will ONLY be used for finetuning
        optim_load_path = os.path.join(save_path, optim_filename)
        if os.path.exists(optim_load_path):
            try:
                optim_state = torch.load(optim_load_path, map_location=device)
                optimizer.load_state_dict(optim_state)
                print(f"Loaded optimizer from {optim_filename}")
            except Exception as e:
                print(e)
                print(f"Optimizer loading failed. Starting training from initial optimizer settings...")

        # Load model state dict (either best or latest)
        save_filename = f'{continue_train}_net.pth'
        load_path = os.path.join(save_path, save_filename)
        if os.path.exists(load_path):
            print('loading the model from %s' % load_path)
            state_dict = torch.load(load_path, map_location=device)
            mlp.load_state_dict(state_dict)
        else:
            print(f"No saved model {load_path}")

        # Loading training/validation history (lists)
        train_history_file = f'{continue_train}_train_history.pkl'
        with open(os.path.join(save_path, train_history_file), 'rb') as f:
            train_history = pickle.load(f)

        val_history_file = f'{continue_train}_train_history.pkl'
        with open(os.path.join(save_path, val_history_file), 'rb') as f:
            val_history = pickle.load(f)
        best_val_loss = min(val_history)

    mlp.train()
    for epoch in tqdm(range(train_epochs)):
        # Train
        for batch_idx, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = mlp(x)
            loss = lossfcn(y_pred, y)
            loss.backward()
            optimizer.step()

            train_history.append(loss.item())

        # Validate
        if epoch % val_interval == 0:
            mlp.eval()
            with torch.no_grad():
                val_loss = 0
                for batch_idx, (x, y) in enumerate(val_dataloader):
                    x, y = x.to(device), y.to(device)
                    y_pred = mlp(x)
                    val_loss += lossfcn(y_pred, y).item()
                val_loss /= len(val_dataloader)
                print('Epoch: {} \tValidation Loss: {:.6f}'.format(epoch, val_loss))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = mlp.state_dict()
                    torch.save(best_model, os.path.join(save_path, 'best_net.pth'))
                val_history.append(val_loss)

        # Save latest stats
        torch.save(mlp.state_dict(), os.path.join(save_path, 'latest_net.pth'))
        torch.save(optimizer.state_dict(), os.path.join(save_path, 'latest_optim.pth'))
        with open(os.path.join(save_path, 'train_history.pkl'), 'wb') as f:
            pickle.dump(train_history, f)
        with open(os.path.join(save_path, 'val_history.pkl'), 'wb') as f:
            pickle.dump(val_history, f)

    # Testing
    if test_model == 'best':
        mlp.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth')))
    elif test_model == 'latest':
        mlp.load_state_dict(torch.load(os.path.join(save_path, 'latest_net.pth')))

    mlp.eval()
    with torch.no_grad():
        test_loss = 0
        for batch_idx, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = mlp(x)
            test_loss += lossfcn(y_pred, y).item()
        test_loss /= len(test_dataloader)
        print(f'Test Loss ({test_model} model): {test_loss:.6f}')

    return mlp, train_history, val_history, test_loss

if __name__ == "__main__":
    input_d = 100
    hidden_d = [100, 100]
    output_d = 1
    dropout = True

    mlp = MLP(input_d, hidden_d, output_d, dropout=dropout)
    print(mlp)

    # Generate random data
    x = torch.randn(1000, input_d)
    weights = torch.randn(input_d, output_d)
    y = torch.matmul(x, weights)

    # Dataloaders
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    batch_size = 10

    train_size = int(train_ratio * len(x))
    val_size = int(val_ratio * len(x))
    test_size = len(x) - train_size - val_size

    train_x, train_y = x[:train_size], y[:train_size]
    val_x, val_y = x[train_size:train_size+val_size], y[train_size:train_size+val_size]
    test_x, test_y = x[train_size+val_size:], y[train_size+val_size:]

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train
    train_epochs = 100
    val_interval = 1
    lr = 0.001
    loss = 'l1'
    device = 'cpu'
    test_model = 'best'
    save_path = "./outputs/test"
    continue_train = False

    mlp, train_history, val_history, test_loss = train(mlp, train_dataloader, val_dataloader, test_dataloader,
                                                       train_epochs=train_epochs, val_interval=val_interval,
                                                       lr=lr, loss=loss, device=device, test_model=test_model,
                                                       save_path=save_path, continue_train=continue_train)