import argparse
import time
import json
from pathlib import Path
import logging.config
import numpy as np
import torch
from collections import Counter
from torch import optim
from torch.utils.data import DataLoader
from tqdm import trange
from utils.training_utils import init_classifier, get_datasets
from utils.logger_config import create_config
from torch.nn.functional import cross_entropy
from sklearn.metrics import f1_score, classification_report


def get_class_weights(training_data):
    """
    Returns balanced class weights for each class in training with the following formula:
        #wj = n_samples / (n_classes * n_samplesj)
    wj = weight for class j
    """
    labels = training_data.labels
    freqs = Counter(labels)
    n_samples = len(labels)
    n_classes = len(set(labels))
    weight_dic = {}
    for label, freq in freqs.items():
        weight = n_samples / (n_classes * freq)
        weight_dic[label] = weight
    return weight_dic


def train(config, train_loader, valid_loader, model_path, device, classweights):
    """
        method to pretrain a composition model
        :param config: config json file
        :param train_loader: dataloader torch object with training data
        :param valid_loader: dataloader torch object with validation data
        :return: the trained model
        """
    model = init_classifier(config)
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    current_patience = 0
    tolerance = 0.005
    lowest_loss = float("inf")
    best_epoch = 1
    epoch = 1
    train_loss = 0.0
    max_f1 = 0
    early_stopping_criterion = config["validation_metric"]
    print(early_stopping_criterion)
    c_weights = torch.from_numpy(np.array(list(classweights.values())))
    print(c_weights)
    for epoch in range(1, config["num_epochs"] + 1):
        # training loop over all batches
        model.train()
        # these store the losses and accuracies for each batch for one epoch
        train_losses = []
        valid_losses = []
        valid_f1 = []
        train_f1 = []
        # for word1, word2, labels in train_loader:
        pbar = trange(len(train_loader), desc='training...', leave=True)
        for batch in train_loader:
            pbar.update(1)
            batch["device"] = device
            out = model(batch).to("cpu")
            loss = cross_entropy(input=out.float(), target=batch["label"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            _, predictions = torch.max(out, 1)
            f1 = f1_score(y_true=batch["label"], y_pred=predictions, average="macro")
            train_f1.append(f1)
            train_losses.append(loss.item())
            # validation loop over validation batches
        model.eval()
        pbar.close()
        pbar = trange(len(train_loader), desc='validation...', leave=True)
        predictions = []
        for batch in valid_loader:
            pbar.update(1)
            batch["device"] = device
            out = model(batch).to("cpu")
            _, predictions = torch.max(out, 1)
            f1 = f1_score(y_true=batch["label"], y_pred=predictions, average="macro")
            valid_f1.append(f1)
            loss = cross_entropy(input=out, target=batch["label"], weight=c_weights.float())
            valid_losses.append(loss.item())

        predictions = np.array(predictions)
        save_predictions(predictions=predictions, path=prediction_path_dev)

        f_macro = np.average(valid_f1)
        f_train = np.average(train_f1)
        pbar.close()

        # calculate average loss and accuracy over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        # stop when f1 score is the highest
        if early_stopping_criterion == "f1":
            if f_macro > max_f1 - tolerance:
                lowest_loss = valid_loss
                max_f1 = f_macro
                best_epoch = epoch
                current_patience = 0
                torch.save(model.state_dict(), model_path)
            else:
                current_patience += 1
            if current_patience > config["patience"]:
                break
        # stop when loss is the lowest
        else:
            if lowest_loss - valid_loss > tolerance:
                lowest_loss = valid_loss
                best_epoch = epoch
                current_patience = 0
                torch.save(model.state_dict(), model_path)
            else:
                current_patience += 1
            if current_patience > config["patience"]:
                break

        logger.info(
            "current patience: %d , epoch %d , train loss: %.5f, train f1 : %.2f, validation loss: %.5f, validation f1 : %.2f" %
            (current_patience, epoch, train_loss, f_train, valid_loss, f_macro))
    logger.info(
        "training finnished after %d epochs, train loss: %.5f, best epoch : %d , best validation loss: %.5f" %
        (epoch, train_loss, best_epoch, lowest_loss))


def predict(test_loader, model, device):
    """
    predicts labels on unseen data (test set)
    :param test_loader: dataloader torch object with test data
    :param model: trained model
    :param config: config: config json file
    :return: predictions for the given dataset, the loss and accuracy over the whole dataset
    """
    test_loss = []
    predictions = []
    model.to(device)
    pbar = trange(len(test_loader), desc='predict...', leave=True)
    for batch in test_loader:
        pbar.update(1)
        batch["device"] = device
        out = model(batch).squeeze().to("cpu")
        _, prediction = torch.max(out, 1)
        prediction = prediction.tolist()
        predictions.append(prediction)
        loss = cross_entropy(input=out, target=batch["label"])
        test_loss.append(loss.item())
    pbar.close()
    predictions = np.array(predictions)
    return predictions, np.average(test_loss)


def save_predictions(predictions, path):
    np.save(file=path, arr=predictions, allow_pickle=True)


def evaluation(predictions, test_data, label_encoder):
    colloc_predictions = []
    free_predictions = []
    colloc_true = []
    free_true = []
    status = test_data.status
    labels = test_data.labels
    for i in range(len(predictions)):
        if status[i] == "free":
            free_predictions.append(predictions[i])
            free_true.append(labels[i])
        else:
            colloc_predictions.append(predictions[i])
            colloc_true.append(labels[i])
    print("collocations : \n")
    colloc_predictions = label_encoder.inverse_transform(colloc_predictions)
    colloc_true = label_encoder.inverse_transform(colloc_true)
    print(classification_report(y_true=colloc_true,
                          y_pred=colloc_predictions, zero_division=0, labels=list(set(colloc_true))))
    logger.info("collocations : \n")
    logger.info(str(classification_report(y_true=colloc_true,
                          y_pred=colloc_predictions, zero_division=0, labels=list(set(colloc_true)))))
    print("free phrases : \n")
    free_predictions = label_encoder.inverse_transform(free_predictions)
    free_true = label_encoder.inverse_transform(free_true)
    print(classification_report(y_true=free_true,
                          y_pred=free_predictions, zero_division=0, labels=list(set(free_true))))
    logger.info(str(classification_report(y_true=free_true,
                          y_pred=free_predictions, zero_division=0, labels=list(set(free_true)))))


if __name__ == "__main__":

    argp = argparse.ArgumentParser()
    argp.add_argument("path_to_config")
    argp = argp.parse_args()

    with open(argp.path_to_config, 'r') as f:  # read in arguments and save them into a configuration object
        config = json.load(f)

    ts = time.gmtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if name is specified choose specified name to save logging file, else use default name
    if config["save_name"] == "":
        save_name = format(
            "%s_%s" % (config["model"]["type"], time.strftime("%Y-%m-%d-%H_%M_%S", ts)))  # change names
    else:
        save_name = format("%s_%s" % (config["save_name"], time.strftime("%Y-%m-%d-%H_%M_%S", ts)))  # change names

    log_file = str(Path(config["logging_path"]).joinpath(save_name + "_log.txt"))  # change location
    model_path = str(Path(config["model_path"]).joinpath(save_name))
    prediction_path_dev = str(Path(config["logging_path"]).joinpath(save_name + "_test_predictions.npy"))
    prediction_path_test = str(Path(config["model_path"]).joinpath(save_name + "_test_predictions.npy"))

    logging.config.dictConfig(create_config(log_file))
    logger = logging.getLogger("train")
    logger.info("Training %s model. \n Logging to %s \n Save model to %s" % (
        config["model"]["type"], log_file, model_path))

    # set random seed
    np.random.seed(config["seed"])

    # read in data...
    dataset_train, dataset_valid, dataset_test = get_datasets(config)
    print("labelsizes")
    print(len(set(dataset_train.labels)))
    print(len(set(dataset_valid.labels)))
    print(len(set(dataset_test.labels)))
    print(set(dataset_train.labels))
    print(set(dataset_valid.labels))
    print(set(dataset_test.labels))
    class_weights = get_class_weights(dataset_train)

    # load data with torch Data Loader
    train_loader = DataLoader(dataset_train,
                              batch_size=config["iterator"]["batch_size"],
                              shuffle=True,
                              num_workers=0)
    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=len(dataset_valid),
                                               shuffle=False,
                                               num_workers=0)

    # load test data in batches
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=len(dataset_test),
                                              shuffle=False,
                                              num_workers=0)

    model = None

    logger.info("%d training batches" % config["iterator"]["batch_size"])
    logger.info("the training data contains %d words" % len(dataset_train))
    logger.info("the validation data contains %d words" % len(dataset_valid))
    logger.info("the test data contains %d words" % len(dataset_test))
    logger.info("training with the following parameter")
    logger.info(config)

    train(config, train_loader, valid_loader, model_path, device, class_weights)
    # test and & evaluate
    logger.info("Loading best model from %s", model_path)
    valid_model = init_classifier(config)
    valid_model.load_state_dict(torch.load(model_path))
    valid_model.eval()

    if valid_model:
        logger.info("generating predictions for test data...")
        valid_predictions, valid_loss = predict(test_loader, valid_model, device)
        valid_predictions = valid_predictions.squeeze()
        print(valid_predictions.shape)
        save_predictions(predictions=valid_predictions, path=prediction_path_dev)
        logger.info("saved predictions to %s" % prediction_path_dev)
        logger.info("test loss: %.5f" % (valid_loss))
        evaluation(valid_predictions, dataset_test, dataset_test.label_encoder)
    else:
        logging.error("model could not been loaded correctly")
