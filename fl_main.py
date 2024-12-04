#!/usr/bin/env python
import os
import random
import json
import pickle
import argparse
import yaml
from json import JSONEncoder
from tqdm import tqdm
import logging
import sys

from fed_baselines.client_base import FedClient
from fed_baselines.client_fedprox import FedProxClient
from fed_baselines.client_scaffold import ScaffoldClient
from fed_baselines.client_fednova import FedNovaClient
from fed_baselines.server_base import FedServer
from fed_baselines.server_scaffold import ScaffoldServer
from fed_baselines.server_fednova import FedNovaServer

from postprocessing.recorder import Recorder
from preprocessing.baselines_dataloader import divide_data_noiid, divide_data_iid
from utils.models import *
from utils.fed_utils import model_decrypt, save_client_weight, cal_and_set_secret_number, \
    generate_and_split_secret_number, model_encrypt, test_accuracy_of_global_model

torch.set_default_dtype(torch.float64)
json_types = (list, dict, str, int, float, bool, type(None))

# logger
logger = logging.getLogger('fl_main')
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('result.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


def save_2_result(config, recorder):
    # Save the results
    if not os.path.exists(config["system"]["res_root"]):
        os.makedirs(config["system"]["res_root"])

    with open(os.path.join(config["system"]["res_root"], '[\'%s\',' % config["client"]["fed_algo"] +
                                                         '\'%s\',' % config["system"]["model"] +
                                                         str(config["system"]["num_client"]) + ',' +
                                                         str(config["system"]["num_round"]) + ',' +
                                                         str(config["client"]["num_local_epoch"])

                           ) + ']', "w") as jsfile:
        json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)


def fed_args():
    """
    Arguments for running federated learning baselines
    :return: Arguments for federated learning baselines
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True,
                        help='Yaml file for configuration')

    args = parser.parse_args()
    return args


def fed_run():
    """
    Main function for the baselines of federated learning
    """
    args = fed_args()
    with open(args.config, "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)

    algo_list = ["FedAvg", "SCAFFOLD", "FedProx", "FedNova", "Homomorphic"]
    assert config["client"]["fed_algo"] in algo_list, "The federated learning algorithm is not supported"

    dataset_list = ['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', 'CIFAR100']
    assert config["system"]["dataset"] in dataset_list, "The dataset is not supported"

    model_list = ["LeNet", 'AlexCifarNet', "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "CNN", "qCNN"]
    assert config["system"]["model"] in model_list, "The model is not supported"

    np.random.seed(config["system"]["i_seed"])
    torch.manual_seed(config["system"]["i_seed"])
    random.seed(config["system"]["i_seed"])

    client_dict = {}
    recorder = Recorder()

    if config["system"]["iid"]:
        logger.info("data divide is iid")
        trainset_config, testset = divide_data_iid(num_client=config["system"]["num_client"],
                                                   dataset_name=config["system"]["dataset"],
                                                   i_seed=config["system"]["i_seed"])
    else:
        logger.info("data divide is non-iid")
        trainset_config, testset = divide_data_noiid(num_client=config["system"]["num_client"],
                                                     num_local_class=config["system"]["num_local_class"],
                                                     dataset_name=config["system"]["dataset"],
                                                     i_seed=config["system"]["i_seed"])
    max_acc = 0

    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings
    for client_id in trainset_config['users']:
        if config["client"]["fed_algo"] == 'FedAvg':
            client_dict[client_id] = FedClient(client_id, dataset_id=config["system"]["dataset"],
                                               epoch=config["client"]["num_local_epoch"],
                                               model_name=config["system"]["model"],
                                               batch_size=config["client"]["batch_size"],
                                               add_noise=config["client"]["add_noise"],
                                               lr=config["client"]["lr"],
                                               i_seed=config["system"]["i_seed"])
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            client_dict[client_id] = ScaffoldClient(client_id, dataset_id=config["system"]["dataset"],
                                                    epoch=config["client"]["num_local_epoch"],
                                                    model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'FedProx':
            client_dict[client_id] = FedProxClient(client_id, dataset_id=config["system"]["dataset"],
                                                   epoch=config["client"]["num_local_epoch"],
                                                   model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'FedNova':
            client_dict[client_id] = FedNovaClient(client_id, dataset_id=config["system"]["dataset"],
                                                   epoch=config["client"]["num_local_epoch"],
                                                   model_name=config["system"]["model"])
        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])

    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings
    if config["client"]["fed_algo"] == 'FedAvg':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"],
                               model_name=config["system"]["model"])
    elif config["client"]["fed_algo"] == 'SCAFFOLD':
        fed_server = ScaffoldServer(trainset_config['users'], dataset_id=config["system"]["dataset"],
                                    model_name=config["system"]["model"])
        scv_state = fed_server.scv.state_dict()
    elif config["client"]["fed_algo"] == 'FedProx':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"],
                               model_name=config["system"]["model"])
    elif config["client"]["fed_algo"] == 'FedNova':
        fed_server = FedNovaServer(trainset_config['users'], dataset_id=config["system"]["dataset"],
                                   model_name=config["system"]["model"])

    # generate secret number and split it, and distribute share to each client.
    generate_and_split_secret_number(client_dict=client_dict, seed=config["system"]["i_seed"])

    # fed_server.load_testset(testset) # in this system, the model is invisible to the server.

    # initial the fed_server
    fed_server.set_model_shape_dtype()
    global_state_dict = fed_server.state_dict()
    fed_server.generate_pk_and_sk()

    # Main process of federated learning in multiple communication rounds.
    pbar_server_agg = tqdm(range(config["system"]["num_round"] + 1), position=0, leave=True)
    logger.info("Training process start ...")
    for global_round in pbar_server_agg:
        logger.info("global round : %d", global_round)
        accuracy = 0
        pbar_clients = tqdm(trainset_config['users'], position=1, leave=False)

        # select random client to cal global_model accuracy.
        random_client_id = random.choice(trainset_config['users'])
        logger.info("the random client id to test accuracy : %s", random_client_id)
        # keys list to encrypt
        # Construct_LeNet = ['conv1.weight', 'conv1.bias', 'fc3.weight', 'fc3.bias']
        Construct_AlexCifarNet = ['features.0.weight', 'features.0.bias', 'classifier.4.weight', 'classifier.4.bias']
        Construct_CNN = ['conv1.weight','conv1.bias','fc.bias']
        Non_list = []
        keys_to_encrypt = Non_list
        for client_id in pbar_clients:
            if client_id != random_client_id and global_round == config["system"]["num_round"]:
                continue
            logger.info("----- client id [%s] -----", client_id)
            # Local training.
            if config["client"]["fed_algo"] == 'FedAvg':

                # update the client_id's model_state as global_model.
                client_dict[client_id].update(global_state_dict)

                # recover the model with client's secret_numer.
                if global_round != 0:
                    client_dict[client_id].recover_model()

                # cal accuracy of global_model
                if client_id == random_client_id and global_round != 0:
                    accuracy = test_accuracy_of_global_model(client_dict[client_id].model, testset)
                    logger.info("global round [%d] client_dict[%s].accuracy : %f", global_round, client_id,
                                accuracy)

                # the final global round
                if client_id == random_client_id and global_round == config["system"]["num_round"]:
                    accuracy = test_accuracy_of_global_model(client_dict[client_id].model, testset)
                    recorder.res['server']['iid_accuracy'].append(accuracy)
                    save_2_result(config, recorder)
                    logger.info("Final accuracy : %f", accuracy)
                    sys.exit(0)

                # client_dict initial
                client_dict[client_id].set_public_key(fed_server.public_key)
                client_dict[client_id].set_global_epoch(global_round)
                logger.info("client_dict[%s] training ...", client_id)

                # local model train
                state_dict, n_data, loss = client_dict[client_id].train()

                logger.info("client_dict[%s] local model encrypting ...", client_id)

                # encrypt the model with pk
                encrypted_model_state_dict = model_encrypt(state_dict, client_dict[client_id].public_key,
                                                           keys_to_encrypt=keys_to_encrypt)
                logger.info("Server receive client_dict[%s] info ...", client_id)

                # server receive the client_dict[client_id]'s message
                fed_server.rec(client_dict[client_id].name, encrypted_model_state_dict, n_data, loss)

            elif config["client"]["fed_algo"] == 'Homomorphic':
                pass
            elif config["client"]["fed_algo"] == 'SCAFFOLD':
                client_dict[client_id].update(global_state_dict, scv_state)
                state_dict, n_data, loss, delta_ccv_state = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, delta_ccv_state)
            elif config["client"]["fed_algo"] == 'FedProx':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            elif config["client"]["fed_algo"] == 'FedNova':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss, coeff, norm_grad = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, coeff, norm_grad)

        # Global aggregation
        logger.info("global aggregation process ...")

        # server selects clients and saves the weight of each client
        fed_server.select_clients()
        save_client_weight(fed_server.n_data, client_dict, fed_server.selected_clients, fed_server)

        # cal secret number with weight and set to client_dict
        cal_and_set_secret_number(client_dict=client_dict, select_clients=fed_server.selected_clients)

        if config["client"]["fed_algo"] == 'FedAvg':
            # global_state_dict, avg_loss, _ = fed_server.agg()

            # homomorphic encrypted aggregation
            global_state_dict, avg_loss, _ = fed_server.agg_hm_en()

            # decrypt the encrypted global model with server.sk
            global_state_dict = model_decrypt(global_state_dict, fed_server.private_key,
                                              fed_server.model_shape_type, keys_to_encrypt=keys_to_encrypt)
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            global_state_dict, avg_loss, _, scv_state = fed_server.agg()  # scarffold
        elif config["client"]["fed_algo"] == 'FedProx':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'FedNova':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'Homomorphic':
            global_state_dict, avg_loss, _ = fed_server.agg_hm_en()

        # Testing and flushing
        # accuracy = fed_server.test() # in our system, the accuracy is calculated by client.
        fed_server.flush()

        # Record the results
        if global_round != 0:
            recorder.res['server']['iid_accuracy'].append(accuracy)
            logger.info("[global round : %d] aggregated model accuracy : %s", global_round, accuracy)
        recorder.res['server']['train_loss'].append(avg_loss)
        logger.info("[global round : %d] aggregated model loss : %s", global_round, avg_loss)

        if max_acc < accuracy:
            max_acc = accuracy
        pbar_server_agg.set_description(
            'Global Round: %d' % int(global_round + 1) +
            '| Train loss: %.4f ' % avg_loss +
            '| Accuracy: %.4f' % accuracy +
            '| Max Acc: %.4f' % max_acc)

        # Save the results
        if not os.path.exists(config["system"]["res_root"]):
            os.makedirs(config["system"]["res_root"])

        with open(os.path.join(config["system"]["res_root"], '[\'%s\',' % config["client"]["fed_algo"] +
                                                             '\'%s\',' % config["system"]["model"] +
                                                             str(config["system"]["num_client"]) + ',' +
                                                             str(config["system"]["num_round"]) + ',' +
                                                             str(config["client"]["num_local_epoch"])

                               ) + ']', "w") as jsfile:
            json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)


if __name__ == "__main__":
    fed_run()
