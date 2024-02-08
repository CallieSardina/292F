# training gnn models, we already shared trained ones at data/{dataset_name}/gnn

import os
import argparse
import torch
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import numpy as np
import random
from math import floor
from tqdm import tqdm
import torch_geometric

import data


class GNN(torch.nn.Module):

    def __init__(self, num_features, num_classes=2, num_layers=3, dim=20, dropout=0.0):
        super(GNN, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dim = dim
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # First GCN layer.
        self.convs.append(GCNConv(num_features, dim))
        self.bns.append(torch.nn.BatchNorm1d(dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(dim, dim))
            self.bns.append(torch.nn.BatchNorm1d(dim))

        # Fully connected layer.
        self.fc = torch.nn.Linear(dim, num_classes)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, GCNConv):
                m.reset_parameters()
            elif isinstance(m, torch.nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, torch.nn.Linear):
                m.reset_parameters()

    def forward(self, data, edge_weight=None):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # GCNs.
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout after every layer.

        # Pooling and FCs.
        node_embeddings = x
        graph_embedding = global_max_pool(node_embeddings, batch)
        out = self.fc(graph_embedding)
        logits = F.log_softmax(out, dim=-1)

        return node_embeddings, graph_embedding, logits

def train(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    lambda_penalty = 0.05
    # degree_penalty_coeff = 0.1

    for train_batch in tqdm(train_loader, desc='Train Batch', total=len(train_loader)):
        optimizer.zero_grad()

        # Forward pass
        node_embeddings, graph_embedding, logits = model(train_batch)
        probabilities = F.softmax(logits, dim=1)

        # Compute classification loss
        classification_loss = F.nll_loss(logits, train_batch.y)

        # Calculate degree of each node
        degrees = torch_geometric.utils.degree(train_batch.edge_index[0], num_nodes=train_batch.num_nodes)

        # Penalize low-probability classifications
        penalty_term = torch.mean(degrees[train_batch.y] * (1 - probabilities[range(len(train_batch.y)), train_batch.y]))
        penalty_loss = lambda_penalty * penalty_term

        # Total loss
        loss = classification_loss + penalty_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * train_batch.num_graphs

        # # Forward pass
        # node_embeddings, graph_embedding, logits = model(train_batch)
        # probabilities = F.softmax(logits, dim=1)

        # # Cross-entropy loss
        # classification_loss = F.nll_loss(logits, train_batch.y)

        # # Calculate degree of each node
        # degrees = torch_geometric.utils.degree(train_batch.edge_index[0], num_nodes=train_batch.num_nodes)

        # # Define penalty term (e.g., penalize high-degree nodes more)
        # penalty_term = -torch.log(probabilities[range(len(train_batch.y)), train_batch.y])#degree_penalty_coeff * torch.mean(degrees * (1 - torch.softmax(logits, dim=-1).max(dim=-1).values))

        # # Total loss
        # loss = classification_loss + penalty_term
        # # loss.backward()
        # loss.backward()
        # optimizer.step()

        # # total_loss += loss.item() * train_batch.num_graphs
        # total_loss += loss.item() * train_batch.num_graphs

        # # Forward pass
        # node_embeddings, graph_embedding, logits = model(train_batch)
        # probabilities = F.softmax(logits, dim=1)

        # # Compute classification loss
        # classification_loss = F.nll_loss(logits, train_batch.y)

        # # Calculate degree of each node
        # degrees = torch_geometric.utils.degree(train_batch.edge_index[0], num_nodes=train_batch.num_nodes)

        # # Penalize low-probability classifications
        # penalty_term = torch.mean(degrees * (1 - torch.softmax(logits, dim=-1).max(dim=-1).values))#-torch.log(probabilities[range(len(train_batch.y)), train_batch.y])
        # penalty_loss = lambda_penalty * penalty_term.mean()

        # # Total loss
        # loss = classification_loss + penalty_loss

        # # Backward pass and optimization
        # loss.backward()
        # optimizer.step()

        # total_loss += loss.item() * train_batch.num_graphs
        #
        # optimizer.zero_grad()
        # #print(train_batch)
        # logits = model(train_batch)[-1]#model(train_batch.to(device))[-1]
        # loss = F.nll_loss(logits, train_batch.y) # -
        # loss.backward()
        # optimizer.step()
        # total_loss += loss.item() * train_batch.num_graphs

    print("Train accuracy: ", total_loss / len(train_loader.dataset))
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def eval(model, eval_loader, device):
    model.eval()
    total_loss = 0
    total_hits = 0

    for eval_batch in tqdm(eval_loader, desc='Eval Batch', total=len(eval_loader)):
        logits = model(eval_batch.to(device))[-1]
        loss = F.nll_loss(logits, eval_batch.y)
        total_loss += loss.item() * eval_batch.num_graphs
        pred = torch.argmax(logits, dim=-1)
        hits = (pred == eval_batch.y).sum()
        total_hits += hits

    print("Eval accuracy: ", total_loss / len(eval_loader.dataset), total_hits / len(eval_loader.dataset))
    return total_loss / len(eval_loader.dataset), total_hits / len(eval_loader.dataset)


def split_data(dataset, valid_ratio=0.1, test_ratio=0.1):
    valid_size = floor(len(dataset) * valid_ratio)
    test_size = floor(len(dataset) * test_ratio)
    train_size = len(dataset) - valid_size - test_size
    splits = torch.utils.data.random_split(dataset, lengths=[train_size, valid_size, test_size])

    return splits


def load_trained_gnn(dataset_name, loss_penalty, device):
    dataset = data.load_dataset(dataset_name)
    model = GNN(
        num_features=dataset.num_features,
        num_classes=2,
        num_layers=3,
        dim=20,
        dropout=0.0
    ).to(device)
    model.load_state_dict(torch.load(f'data/{dataset_name}/loss_penalty_{loss_penalty}/model_best.pth', map_location=device))
    return model


@torch.no_grad()
def load_trained_prediction(dataset_name, loss_penalty, device):
    prediction_file = f'data/{dataset_name}/loss_penalty_{loss_penalty}/preds.pt'
    if os.path.exists(prediction_file):
        return torch.load(prediction_file, map_location=device)
    else:
        dataset = data.load_dataset(dataset_name)
        model = load_trained_gnn(dataset_name, loss_penalty, device)
        model.eval()

        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        preds = []
        for eval_batch in tqdm(loader, desc='Eval Batch', total=len(loader)):
            logits = model(eval_batch.to(device))[-1]
            pred = torch.argmax(logits, dim=-1)
            preds.append(pred)
        preds = torch.cat(preds)
        torch.save(preds, prediction_file)
        return preds


@torch.no_grad()
def load_trained_embeddings_logits(dataset_name, loss_penalty, device):
    node_embeddings_file = f'data/{dataset_name}/loss_penalty_{loss_penalty}/node_embeddings.pt'
    graph_embeddings_file = f'data/{dataset_name}/loss_penalty_{loss_penalty}/graph_embeddings.pt'
    logits_file = f'data/{dataset_name}/loss_penalty_{loss_penalty}/logits.pt'
    if os.path.exists(node_embeddings_file) and os.path.exists(graph_embeddings_file) and os.path.exists(logits_file):
        node_embeddings = torch.load(node_embeddings_file)
        for i, node_embedding in enumerate(node_embeddings):  # every graph has different size
            node_embeddings[i] = node_embeddings[i].to(device)
        graph_embeddings = torch.load(graph_embeddings_file, map_location=device)
        logits = torch.load(logits_file, map_location=device)
        return node_embeddings, graph_embeddings, logits
    else:
        dataset = data.load_dataset(dataset_name)
        model = load_trained_gnn(dataset_name, loss_penalty, device)
        model.eval()
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        graph_embeddings, node_embeddings, logits = [], [], []
        for eval_batch in tqdm(loader, desc='Eval Batch', total=len(loader)):
            node_emb, graph_emb, logit = model(eval_batch.to(device))
            max_batch_number = max(eval_batch.batch)
            for i in range(max_batch_number + 1):
                idx = torch.where(eval_batch.batch == i)[0]
                node_embeddings.append(node_emb[idx])
            graph_embeddings.append(graph_emb)
            logits.append(logit)
        graph_embeddings = torch.cat(graph_embeddings)
        logits = torch.cat(logits)
        torch.save([node_embedding.cpu() for node_embedding in node_embeddings], node_embeddings_file)
        torch.save(graph_embeddings, graph_embeddings_file)
        torch.save(logits, logits_file)
        return node_embeddings, graph_embeddings, logits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mutagenicity',
                        help="Dataset. Options are ['mutagenicity', 'aids', 'nci1', 'proteins']. Default is 'mutagenicity'. ")
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate. Default is 0.0. ')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size. Default is 128.')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GCN layers. Default is 3.')
    parser.add_argument('--dim', type=int, default=20,
                        help='Number of GCN dimensions. Default is 20. ')
    parser.add_argument('--random-seed', type=int, default=0,
                        help='Random seed for training. Default is 0. ')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs. Default is 1000. ')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate. Default is 0.001. ')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Index of cuda device to use. Default is 0. ')
    parser.add_argument('--loss_penalty', type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    loss_penalty = args.loss_penalty

    # Load and split the dataset.
    dataset = data.load_dataset(args.dataset)

    train_set, valid_set, test_set = split_data(dataset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    # Logging.
    gnn_folder = f'data/{args.dataset}/loss_penalty_{args.loss_penalty}/'
    if not os.path.exists(gnn_folder):
        os.makedirs(gnn_folder)
    log_file = gnn_folder + 'log.txt'
    with open(log_file, 'w') as f:
        pass

    # Initialize the model.
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    model = GNN(
        num_features=10,
        num_classes=2,
        num_layers=args.num_layers,
        dim=args.dim,
        dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training.
    start_epoch = 1
    epoch_iterator = tqdm(range(start_epoch, start_epoch + args.epochs), desc='Epoch')
    best_valid = float('inf')
    best_epoch = 0
    for epoch in epoch_iterator:
        train_loss = train(model, optimizer, train_loader, device)
        valid_loss, valid_acc = eval(model, valid_loader, device)
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), gnn_folder + f'model_checkpoint{epoch}.pth')
            torch.save(optimizer.state_dict(), gnn_folder + f'optimizer_checkpoint{epoch}.pth')
        with open(log_file, 'a') as f:
            print(f'Epoch = {epoch}:', file=f)
            print(f'Train Loss = {train_loss:.4e}', file=f)
            print(f'Valid Loss = {valid_loss:.4e}', file=f)
            print(f'Valid Accuracy = {valid_acc:.4f}', file=f)

    # Testing.
    model.load_state_dict(torch.load(gnn_folder + f'model_checkpoint{best_epoch}.pth', map_location=device))
    torch.save(model.state_dict(), gnn_folder + f'model_best.pth')
    train_acc = eval(model, train_loader, device)[1]
    valid_acc = eval(model, valid_loader, device)[1]
    test_acc = eval(model, test_loader, device)[1]
    with open(log_file, 'a') as f:
        print(file=f)
        print(f"Best Epoch = {best_epoch}:", file=f)
        print(f"Train Accuracy = {train_acc:.4f}", file=f)
        print(f"Valid Accuracy = {valid_acc:.4f}", file=f)
        print(f"Test Accuracy = {test_acc:.4f}", file=f)


if __name__ == '__main__':
    main()
