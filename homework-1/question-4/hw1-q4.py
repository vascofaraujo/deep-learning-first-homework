#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

import utils


# Q3.1
class LogisticRegression(nn.Module):

    def __init__(self, n_classes, n_features, **kwargs):
        """
        n_classes (int)
        n_features (int)

        The __init__ should be used to declare what kind of layers and other
        parameters the module has. For example, a logistic regression module
        has a weight matrix and bias vector. For an idea of how to use
        pytorch to make weights and biases, have a look at
        https://pytorch.org/docs/stable/nn.html
        """
        super().__init__()
        # In a pytorch module, the declarations of layers needs to come after
        # the super __init__ line, otherwise the magic doesn't work.

        self.linear = nn.Linear(n_features, n_classes)


    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples

        Every subclass of nn.Module needs to have a forward() method. forward()
        describes how the module computes the forward pass. In a log-lineear
        model like this, for example, forward() needs to compute the logits
        y = Wx + b, and return y (you don't need to worry about taking the
        softmax of y because nn.CrossEntropyLoss does that for you).

        One nice thing about pytorch is that you only need to define the
        forward pass -- this is enough for it to figure out how to do the
        backward pass.
        """

        y = self.linear(x)

        return y




# Q3.2
class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """
        n_classes (int)
        n_features (int)
        hidden_size (int)
        layers (int)
        activation_type (str)
        dropout (float): dropout probability

        As in logistic regression, the __init__ here defines a bunch of
        attributes that each FeedforwardNetwork instance has. Note that nn
        includes modules for several activation functions and dropout as well.
        """
        super().__init__()
        # Implement me!
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.activation = nn.ReLU() if activation_type == 'relu' else nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.extra_layers = layers - 1
        self.fc2 = nn.Linear(hidden_size, n_classes)


    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples

        This method needs to perform all the computation needed to compute
        the output logits from x. This will include using various hidden
        layers, pointwise nonlinear functions, and dropout.
        """
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.activation(x)
        for _ in range(self.extra_layers):
            print("adding extra layer")
            x = self.fc_hidden(x)
            x = self.dropout(x)
            x = self.activation(x)
        y = self.fc2(x)

        return y


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function



    Check out https://pytorch.org/docs/stable/optim.html for examples of how
    to use an optimizer object to update the parameters.

    This function should return the loss (tip: call loss.item()) to get the
    loss as a numerical value that is not part of the computation graph.
    """
    optimizer.zero_grad()

    loss = criterion(model(X), y)

    loss.backward()
    optimizer.step()

    return loss.item()




def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def tune_hyperparameters(dataset, opt, model):
    possible_lr = [0.001, 0.01, 0.1]
    possible_hidden_size = [100, 200]
    possible_dropout = [0.3, 0.5]
    possible_activation = ['relu', 'tanh']
    possible_optimizer = ['sgd', 'adam']

    default_hyperparameters = {'lr': opt.learning_rate,
                            'hidden_size': opt.hidden_sizes,
                            'dropout': opt.dropout,
                            'activation': opt.activation,
                            'optimizer': opt.optimizer}
    print(default_hyperparameters)

    acc_lr = []
    print("Tuning learning rate...\n")
    for index, lr in enumerate(possible_lr):
        print(f"Loop {index+1}/{len(possible_lr)}")
        opt.learning_rate = lr
        test_acc = train_model(dataset, opt, model)
        acc_lr.append(test_acc)
    best_lr = possible_lr[acc_lr.index(max(acc_lr))]
    print(f"Best learning rate: {best_lr} with an acc of {max(acc_lr)}")
    opt.learning_rate = default_hyperparameters['lr']

    acc_hidden_size = []
    print("Tuning hidden size...\n")
    for index, hidden_size in enumerate(possible_hidden_size):
        print(f"Loop {index+1}/{len(possible_hidden_size)}")
        opt.hidden_size = hidden_size
        test_acc = train_model(dataset, opt, model)
        acc_hidden_size.append(test_acc)
    best_hidden_size = possible_hidden_size[acc_hidden_size.index(max(acc_hidden_size))]
    print(f"Best hidden size: {best_hidden_size} with an acc of {max(acc_hidden_size)}")
    opt.hidden_sizes = default_hyperparameters['hidden_size']

    acc_dropout = []
    print("Tuning dropout...\n")
    for index, dropout in enumerate(possible_dropout):
        print(f"Loop {index+1}/{len(possible_dropout)}")
        opt.dropout = dropout
        test_acc = train_model(dataset, opt, model)
        acc_dropout.append(test_acc)
    best_dropout = possible_dropout[acc_dropout.index(max(acc_dropout))]
    print(f"Best dropout: {best_dropout} with an acc of {max(acc_dropout)}")
    opt.dropout = default_hyperparameters['dropout']

    acc_activation = []
    print("Tuning activation...\n")
    for index, activation in enumerate(possible_activation):
        print(f"Loop {index+1}/{len(possible_activation)}")
        opt.activation = activation
        test_acc = train_model(dataset, opt, model)
        acc_activation.append(test_acc)
    best_activation = possible_activation[acc_activation.index(max(acc_activation))]
    print(f"Best activation: {best_activation} with an acc of {max(acc_activation)}")
    opt.activation = default_hyperparameters['activation']

    acc_optimizer = []
    print("Tuning optimizer...\n")
    for index, optimizer in enumerate(possible_optimizer):
        print(f"Loop {index+1}/{len(possible_optimizer)}")
        opt.optimizer = optimizer
        test_acc = train_model(dataset, opt, model)
        acc_optimizer.append(test_acc)
    best_optimizer = possible_optimizer[acc_optimizer.index(max(acc_optimizer))]
    print(f"Best optimizer: {best_optimizer} with an acc of {max(acc_optimizer)}")
    opt.optimizer = default_hyperparameters['optimizer']

    best_hyperparameters = {'lr': best_lr,
                            'hidden_size': best_hidden_size,
                            'dropout': best_dropout,
                            'activation': best_activation,
                            'optimizer': best_optimizer}

    return best_hyperparameters

def train_model(dataset, opt, model):
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)


    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay)

    criterion = nn.CrossEntropyLoss()


    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))

        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))


    #plot(epochs, train_mean_losses, ylabel='Loss', name='training-loss')
    #plot(epochs, valid_accs, ylabel='Accuracy', name='validation-accuracy')

    final_test_acc = evaluate(model, test_X, test_y)
    print('Final Test acc: %.4f' % (final_test_acc))

    return final_test_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=1, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-hidden_sizes', type=int, default=200)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_classification_data()
    dataset = utils.ClassificationDataset(data)

    n_classes = torch.unique(dataset.y).shape[0]  # 10
    n_feats = dataset.X.shape[1]

    if opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
        print("Training Logistic regression ...\n")
        _ = train_model(dataset, opt, model)
    else:
        model = FeedforwardNetwork(
            n_classes, n_feats,
            opt.hidden_sizes, opt.layers,
            opt.activation, opt.dropout)
        print("Training Feed Forward Network...\n")
        best_hyperparameters = tune_hyperparameters(dataset, opt, model)
        print(best_hyperparameters)


if __name__ == '__main__':
    main()
