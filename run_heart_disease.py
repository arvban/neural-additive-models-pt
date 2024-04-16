import os
import tqdm
import copy
import random
import logging
from absl import app
from absl import flags
from torch.utils.data import TensorDataset, DataLoader

import nam.metrics
from nam.metrics import compute_classification_metrics
import nam.data_utils
from nam.model import *

FLAGS = flags.FLAGS

flags.DEFINE_integer("training_epochs", 20, "The number of epochs to run training for.")
flags.DEFINE_float("learning_rate", 1e-2, "Hyperparameter: learning rate.")
flags.DEFINE_float("output_regularization", 0.0, "Hyperparameter: feature reg")
flags.DEFINE_float("l2_regularization", 0.0, "Hyperparameter: l2 weight decay")
flags.DEFINE_integer("batch_size", 16, "Hyperparameter: batch size.")
flags.DEFINE_string("log_file", None, "File where to store summaries.")
flags.DEFINE_string("dataset", "BreastCancer", "Name of the dataset to load for training.")
flags.DEFINE_float("decay_rate", 0.995, "Hyperparameter: Optimizer decay rate")
flags.DEFINE_float("dropout", 0.1, "Hyperparameter: Dropout rate")
flags.DEFINE_integer("data_split", 1, "Dataset split index to use. Possible values are 1 to `FLAGS.num_splits`.")
flags.DEFINE_integer("seed", 1, "Seed used for reproducibility.")
flags.DEFINE_float("feature_dropout", 0.1, "Hyperparameter: Prob. with which features are dropped")
flags.DEFINE_integer("n_basis_functions", 1000, "Number of basis functions to use in a FeatureNN for a real-valued feature.")
flags.DEFINE_integer("units_multiplier", 4, "Number of basis functions for a categorical feature")
flags.DEFINE_integer("n_models", 1, "the number of models to train.")
flags.DEFINE_integer("n_splits", 3, "Number of data splits to use")
flags.DEFINE_integer("id_fold", 1, "Index of the fold to be used")
flags.DEFINE_list("hidden_units", [64, 32], "Amounts of neurons for additional hidden layers, e.g. 64,32,32")
flags.DEFINE_string("shallow_layer", "relu", "Activation function used for the first layer: (1) relu, (2) exu")
flags.DEFINE_string("hidden_layer", "relu", "Activation function used for the hidden layers: (1) relu, (2) exu")
flags.DEFINE_boolean("regression", False, "Boolean for regression or classification")
flags.DEFINE_integer("early_stopping_epochs", 10, "Early stopping epochs")
_N_FOLDS = 5


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(x_train, y_train, x_validate, y_validate, device):
    model = NeuralAdditiveModel(
        input_size=x_train.shape[-1],
        shallow_units=nam.data_utils.calculate_n_units(x_train, FLAGS.n_basis_functions, FLAGS.units_multiplier),
        hidden_units=list(map(int, FLAGS.hidden_units)),
        shallow_layer=ExULayer if FLAGS.shallow_layer == "exu" else ReLULayer,
        hidden_layer=ExULayer if FLAGS.hidden_layer == "exu" else ReLULayer,
        hidden_dropout=FLAGS.dropout,
        feature_dropout=FLAGS.feature_dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=FLAGS.learning_rate,
                                  weight_decay=FLAGS.l2_regularization)
    criterion = nam.metrics.penalized_mse if FLAGS.regression else nam.metrics.penalized_cross_entropy
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    validate_dataset = TensorDataset(torch.tensor(x_validate), torch.tensor(y_validate))
    validate_loader = DataLoader(validate_dataset, batch_size=FLAGS.batch_size, shuffle=True)

    n_tries = FLAGS.early_stopping_epochs
    best_validation_score, best_weights = 0, None

    for epoch in range(FLAGS.training_epochs):
        model.train()
        total_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        logging.info(f"epoch {epoch} | train | loss = {total_loss:.5f}")

        scheduler.step()

        model.eval()
        validation_loss, val_score = evaluate(model, validate_loader, criterion, device)
        logging.info(f"epoch {epoch} | validate | loss = {validation_loss:.5f} | metric = {val_score:.5f}")

        # early stopping
        if val_score <= best_validation_score and n_tries > 0:
            n_tries -= 1
            continue
        elif val_score <= best_validation_score:
            logging.info("Early stopping at epoch {epoch}")
            break
        best_validation_score = val_score
        best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model

def train_one_epoch(model, criterion, optimizer, data_loader, device):
    total_loss = 0
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        logits, fnns_out = model.forward(x)
        loss = criterion(logits, y, fnns_out, feature_penalty=FLAGS.output_regularization)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    return average_loss

def evaluate(model, data_loader, criterion, device):
    total_loss = 0
    total_score = 0
    metric = None
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits, fnns_out = model.forward(x)
            loss = criterion(logits, y, fnns_out, feature_penalty=FLAGS.output_regularization)
            total_loss += loss.item()
            metric, score = nam.metrics.calculate_metric(logits, y, regression=FLAGS.regression)
            total_score += score
    average_loss = total_loss / len(data_loader)
    average_score = total_score / len(data_loader)
    return average_loss, average_score



import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import gaussian_kde

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import gaussian_kde

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch


def plot_feature_functions(model, dataset_X, feature_stats, bins=30):
    feature_names = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 
                     'Sex_F', 'Sex_M', 
                     'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 
                     'FastingBS_0', 'FastingBS_1', 'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST', 
                     'ExerciseAngina_N', 'ExerciseAngina_Y', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']
    num_features = len(feature_names)
    plt.figure(figsize=(14, 20))

    for i, feature in enumerate(feature_names):
        ax = plt.subplot((num_features + 2) // 3, 3, i + 1)
        ylim_original = ax.get_ylim()  # Get the current y-limits to use in the fill

        # Unstandardize data if applicable
        if feature in feature_stats:
            mean = feature_stats[feature]['mean']
            std = feature_stats[feature]['std']
            data = dataset_X[:, i].numpy() * std + mean
        else:
            data = dataset_X[:, i].numpy()


        # Generate evaluation points for the feature
        if feature in ['Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA', 
                       'ChestPainType_NAP', 'ChestPainType_TA', 'FastingBS_0', 'FastingBS_1', 
                       'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_N', 
                       'ExerciseAngina_Y', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']:
            X_eval = torch.zeros((2, num_features))
            X_eval[:, i] = torch.tensor([0, 1], dtype=torch.float32)
        else:
            min_val = data.min()
            max_val = data.max()
            x_range = np.linspace(min_val, max_val, 100)
            X_eval = torch.zeros((100, num_features), dtype=torch.float32)
            X_eval[:, i] = torch.from_numpy(x_range)

        # Pass data through the model
        with torch.no_grad():
            _, fnns_out = model(X_eval)

        # Plotting model outputs
        if feature in feature_names[-16:]:
            ax.scatter([0, 1], fnns_out[:, i].numpy(), color='blue', alpha=1.0)
            ax.set_xlim(-0.5, 1.5)
        else:
            ax.plot(x_range, fnns_out[:, i].numpy(), color='blue', alpha=1.0)

        ylim = ax.get_ylim()  # Get the current y-limits to use in the fill

        # Density plot for continuous features and bar shading for categorical features
        if feature in feature_names[-16:]:
            # Categorical variables shading

            counts, bin_edges = np.histogram(data, bins=2, density=False)
            max_count = counts.max()
            for count, edge_left, edge_right in zip(counts, bin_edges[:-1], bin_edges[1:]):
                ax.fill_betweenx(ylim, edge_left, edge_right, color='red', alpha=count/(5*max_count))


        else:
            # Continuous variables density shading
            counts, bin_edges = np.histogram(data, bins=bins, density=False)
            max_count = counts.max()
            for count, edge_left, edge_right in zip(counts, bin_edges[:-1], bin_edges[1:]):
                ax.fill_betweenx(ylim, edge_left, edge_right, color='red', alpha=count/(2*max_count))


        # Set axis labels and titles
        ax.set_title(feature)
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('FNN Output')

        # Create custom legend handles
        legend_handles = [
            Patch(color='blue', label='FNN Output'),
            Patch(color='pink', label='Data Distribution')
        ]

        # Add the custom legend to the plot
        ax.legend(handles=legend_handles)

    plt.tight_layout()
    plt.savefig('plot_NaM.png')
    plt.show()







def main(args):
    # Set up logging
    seed_everything(FLAGS.seed)

    handlers = [logging.StreamHandler()]
    if FLAGS.log_file:
        handlers.append(logging.FileHandler(FLAGS.log_file))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        handlers=handlers)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("load data")
    
    # Load dataset
    column_names = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'Sex_F', 'Sex_M', 
                    'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 
                    'FastingBS_0', 'FastingBS_1', 'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST', 
                    'ExerciseAngina_N', 'ExerciseAngina_Y', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']

    X_train = pd.read_csv('/Users/arvid/Documents/neural-additive-models-pt/X_train_preprocessed_2.csv', header=0, skiprows=1, names=column_names)
    X_test = pd.read_csv('/Users/arvid/Documents/neural-additive-models-pt/X_test_preprocessed_2.csv', header=0, skiprows=1, names=column_names)

    y_train = pd.read_csv('/Users/arvid/Documents/neural-additive-models-pt/y_train_2.csv')
    y_test = pd.read_csv('/Users/arvid/Documents/neural-additive-models-pt/y_test_2.csv')

    # Split train into train and validation
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.4, random_state=FLAGS.seed)


    # Create tensors for the training data
    train_dataset_X = torch.tensor(X_train.values, dtype=torch.float32)
    train_dataset_y = torch.tensor(y_train.values.ravel(), dtype=torch.float32)

    # Create tensors for the validation data
    validate_dataset_X = torch.tensor(X_validate.values, dtype=torch.float32)
    validate_dataset_y = torch.tensor(y_validate.values.ravel(), dtype=torch.float32)


    # # Convert to PyTorch tensors and create loaders
    # train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values.ravel(), dtype=torch.float32))
    # validate_dataset = TensorDataset(torch.tensor(X_validate.values, dtype=torch.float32), torch.tensor(y_validate.values.ravel(), dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values.ravel(), dtype=torch.float32))


    # train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    # validate_loader = DataLoader(validate_dataset, batch_size=FLAGS.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    logging.info("begin training")

    model = train_model(train_dataset_X , train_dataset_y, validate_dataset_X, validate_dataset_y, device)

    logging.info("training complete")

    # Evaluate on test set
    logging.info("evaluating on test set")
    criterion = nam.metrics.penalized_mse if FLAGS.regression else nam.metrics.penalized_cross_entropy

    test_metric, test_score = evaluate(model, test_loader, criterion, device)
    logging.info(f"test evaluation | {test_metric}={test_score}")



    # Model evaluation on test data
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    logging.info(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Collect all logits and truths from the test set for additional metrics
    all_logits = []
    all_truths = []
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            logits, _ = model(x)  # Assuming model returns logits and some other outputs
            all_logits.append(logits)
            all_truths.append(y)

    # Concatenate all logits and truths
    all_logits = torch.cat(all_logits)
    all_truths = torch.cat(all_truths)

    # Compute additional classification metrics
    additional_metrics = compute_classification_metrics(all_logits, all_truths)
    logging.info("Additional Classification Metrics:")
    for metric, value in additional_metrics.items():
        if isinstance(value, np.ndarray):
            logging.info(f"{metric}:\n{value}")
        else:
            logging.info(f"{metric}: {value}")






    # reading in original data to undo standardization
        # Load dataset
    column_names_original = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope", "HeartDisease"]

    x_original = pd.read_csv('/Users/arvid/Documents/neural-additive-models-pt/train_val_split.csv', header=0, skiprows=1, names=column_names_original)

    # Calculate mean and standard deviation for the continuous features
    feature_stats = {
        'Age': {'mean': x_original['Age'].mean(), 'std': x_original['Age'].std()},
        'RestingBP': {'mean': x_original['RestingBP'].mean(), 'std': x_original['RestingBP'].std()},
        'Cholesterol': {'mean': x_original['Cholesterol'].mean(), 'std': x_original['Cholesterol'].std()},
        'MaxHR': {'mean': x_original['MaxHR'].mean(), 'std': x_original['MaxHR'].std()},
        'Oldpeak': {'mean': x_original['Oldpeak'].mean(), 'std': x_original['Oldpeak'].std()}
    }



    plot_feature_functions(model, train_dataset_X, feature_stats)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    app.run(main)