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


def tune_hyperparameters(dataset, opt, n_classes, n_feats, parameter, possible_hyperparameters, is_logistic):

    acc_parameter = []
    print(f"Tuning {parameter}...\n")
    for index, curr_parameter in enumerate(possible_hyperparameters[parameter]):
        print(f"Loop {index+1}/{len(possible_hyperparameters[parameter])}")
        print(curr_parameter)
        if parameter == 'lr':
            opt.learning_rate = curr_parameter
        elif parameter == 'hidden_size':
            opt.hidden_sizes = curr_parameter
        elif parameter == 'dropout':
            opt.dropout = curr_parameter
        elif parameter == 'activation':
            opt.activation = curr_parameter
        elif parameter == 'optimizer':
            opt.optimizer = curr_parameter

        test_acc = train_model(dataset, opt, n_classes, n_feats, False, is_logistic)
        acc_parameter.append(test_acc)
    best_parameter = possible_hyperparameters[parameter][acc_parameter.index(max(acc_parameter))]
    print(f"Best {parameter}: {best_parameter} with an acc of {max(acc_parameter)}")

    return best_parameter

def train_model(dataset, opt, n_classes, n_feats, make_plot, is_logistic):
    if (is_logistic):
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = FeedforwardNetwork(
            n_classes, n_feats,
            opt.hidden_sizes, opt.layers,
            opt.activation, opt.dropout)

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


    final_test_acc = evaluate(model, test_X, test_y)
    print('Final Test acc: %.4f\n' % (final_test_acc))

    if make_plot:
        plot(epochs, train_mean_losses, ylabel='Loss', name='training-loss')
        plot(epochs, valid_accs, ylabel='Accuracy', name='validation-accuracy')

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
    parser.add_argument('-tune',
                        choices=['yes', 'no'], default='no')
    # Added by me

    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_classification_data()
    dataset = utils.ClassificationDataset(data)

    n_classes = torch.unique(dataset.y).shape[0]  # 10
    n_feats = dataset.X.shape[1]

    possible_hyperparameters = {'lr': [0.001, 0.01, 0.1],
                                'hidden_size': [100, 200],
                                'dropout': [0.3, 0.5],
                                'activation': ['relu', 'tanh'],
                                'optimizer': ['sgd', 'adam']
                                }

    if opt.model == 'logistic_regression':
        print("Training Logistic regression ...\n")

        if opt.tune == 'yes':
            opt.learning_rate = tune_hyperparameters(dataset, opt, n_classes, n_feats, 'lr', possible_hyperparameters, True)


        _ = train_model(dataset, opt, n_classes, n_feats, True, True)
    else:
        print("Training Feed Forward Network...\n")

        if opt.tune == 'yes':
            best_lr = tune_hyperparameters(dataset, opt, n_classes, n_feats, 'lr', possible_hyperparameters, False)
            best_hidden_size = tune_hyperparameters(dataset, opt, n_classes, n_feats, 'hidden_size', possible_hyperparameters, False)
            best_dropout = tune_hyperparameters(dataset, opt, n_classes, n_feats, 'dropout', possible_hyperparameters, False)
            best_activation = tune_hyperparameters(dataset, opt, n_classes, n_feats, 'activation', possible_hyperparameters, False)
            best_optimizer = tune_hyperparameters(dataset, opt, n_classes, n_feats, 'optimizer', possible_hyperparameters, False)

            opt.learning_rate = best_lr
            opt.hidden_sizes = best_hidden_size
            opt.dropout = best_dropout
            opt.activation = best_activation
            opt.optimizer = best_optimizer

        _ = train_model(dataset, opt, n_classes, n_feats, True, False)


if __name__ == '__main__':
    main()
