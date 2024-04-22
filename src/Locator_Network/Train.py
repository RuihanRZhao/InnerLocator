import json
from urllib.request import urlopen

import torch
import torch.nn as nn
import torch.optim as optim


def get_environment_data(type):
    response = urlopen(f'http://127.0.0.1:5000/train/{type}/all')

    if response.getcode() == 200:
        data = json.loads(response.read().decode('utf-8'))
        return data
    else:
        print("Error code: " + str(response.getcode()))


def get_source():
    origin_Source_list = get_environment_data('relationship')
    Measured_Source_list = [
        [
            ["from", each["from"], ],
            ["from type", each["from_type"], ],
            ["to", each["to"], ],
            ["to type", each["to_type"], ],
            ["ToA", each["ToA"], ],
            ["AoA theta", each["AoA"]["theta"] if each["AoA"] is not None else None],
            ["AoA phi", each["AoA"]["theta"] if each["AoA"] is not None else None],
        ] for each in origin_Source_list
    ]
    return Measured_Source_list


def get_target():
    origin_Target_list = get_environment_data('location')
    Measured_Target_list = [
        [
            ["name", each["name"]],
            ["pos_x", each["location"][0]],
            ["pos_y", each["location"][1]],
            ["pos_z", each["location"][2]],
        ] for each in origin_Target_list
    ]
    return Measured_Target_list


def save_checkpoint(state, filename="model_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename="model_checkpoint.pth"):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train_and_validate(model, criterion, optimizer, device, num_epochs, filename="model_checkpoint.pth"):
    model.to(device)
    model.train()
    best_acc = 0  # Track the best accuracy

    for epoch in range(num_epochs):
        # Forward pass
        sources = get_source()
        targets = get_target()
        answers, name_index = model(sources, [item[0][1] for item in targets], len(targets))
        for i, item in enumerate(targets):
            item[0] = name_index[i]
            item[1] = item[1][1]
            item[2] = item[2][1]
            item[3] = item[3][1]

        targets_tensor = torch.tensor(targets).to(device)

        loss = criterion(answers, targets_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation phase
        # val_acc = validate(model, val_loader, criterion)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f},"
              # f" Val Acc: {val_acc:.2f}%"
              )

        # Save the model if there is improvement in accuracy
        # if val_acc > best_acc and val_acc >= best_acc + 5:
        #     best_acc = val_acc
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'accuracy': val_acc,
        #     }, filename)
        #     print(f"Checkpoint saved with improvement: {val_acc:.2f}%")


def validate(model, data_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            output_target_num = targets.size(0)
            outputs = model(inputs, output_target_num)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    from Network import InnerLocatorNetwork

    device = torch.device("cpu" if torch.cuda.is_available() else "")

    initial_vocab_size = 0
    initial_vocab_list = []
    vocab_embedding_dim = 1
    feature_dim = 1
    num_layers_Transformer = 2
    num_heads_Transformer = 1
    hidden_dim_Transformer = 32
    hidden_dim_LSTM = 1
    num_layers_LSTM = 1

    brain = InnerLocatorNetwork(
        initial_vocab_size,
        initial_vocab_list,
        vocab_embedding_dim,
        feature_dim,
        num_layers_Transformer,
        num_heads_Transformer,
        hidden_dim_Transformer,
        hidden_dim_LSTM,
        num_layers_LSTM
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(brain.parameters(), lr=0.001)

    # Get data loaders

    # Train and validate the model
    train_and_validate(brain, criterion, optimizer, device, num_epochs=100_000_000)
