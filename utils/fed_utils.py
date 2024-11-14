import random
from utils.models import *
from copy import deepcopy
import math
import logging
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

logger = logging.getLogger('fed_utils')
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('result.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def assign_dataset(dataset_name):
    """
    Assign the parameters to a dataset
    :param dataset_name: Dataset name
    :return: num_class: Number of classes in the dataset
    :return: image_dim: Image dimensions
    :return: image_channel: Number of image channels
    """
    num_class = -1
    image_dim = -1
    image_channel = -1

    if dataset_name == 'MNIST':
        num_class = 10
        image_dim = 28
        image_channel = 1

    elif dataset_name == 'FashionMNIST':
        num_class = 10
        image_dim = 28
        image_channel = 1

    elif dataset_name == 'EMNIST':
        num_class = 27
        image_dim = 28
        image_channel = 1

    elif dataset_name == 'CIFAR10':

        num_class = 10
        image_dim = 32
        image_channel = 3

    elif dataset_name == 'CIFAR100':

        num_class = 100
        image_dim = 32
        image_channel = 3

    elif dataset_name == 'SVHN':

        num_class = 10
        image_dim = 32
        image_channel = 3

    elif dataset_name == 'IMAGENET':

        num_class = 200
        image_dim = 64
        image_channel = 3

    return num_class, image_dim, image_channel


def init_model(model_name, num_class, image_channel):
    """
    Initialize the model for a specific learning task.
    :param model_name: Model name
    :param num_class: Number of classes in the dataset
    :param image_channel: Number of image channels
    :return: The initialized model
    """
    model = None
    if model_name == "ResNet18":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "ResNet50":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "ResNet34":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "ResNet101":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "ResNet152":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "LeNet":
        model = LeNet(num_classes=num_class, in_channels=image_channel)
    elif model_name == "CNN":
        model = CNN(num_classes=num_class, in_channels=image_channel)
    elif model_name == "VGG11":
        model = generate_vgg(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "VGG11_bn":
        model = generate_vgg(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "AlexCifarNet":
        model = AlexCifarNet()
    else:
        print('Model is not supported')

    return model


def gaussian_noise(data_shape, s, sigma, generator, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * s, data_shape, generator=torch.manual_seed(generator)).to(device)


def model_encrypt(ori_model_state_dict, pub_key, keys_to_encrypt):
    """
    this method is to encrypt the model with Paillier encryption, use pub_key and choose several keys to encrypt.
    :param ori_model_state_dict: ori_model.state_dict()
    :param pub_key: Paillier public key
    :param keys_to_encrypt: List of keys of model to encrypt
    :return: encrypted model, encrypted_model_state_dict[key]=(encrypted_list, meta_data{'shape','dtype'})
    """
    encrypted_state_dict = {}
    state_dict = ori_model_state_dict
    for key in state_dict.keys():
        start_time = time.time()
        # encrypt keys in list keys_to_encrypt[].
        if key in keys_to_encrypt:
            encrypted_list = []
            list_w = state_dict[key].view(-1).cpu().tolist()
            if any(math.isnan(x) for x in list_w):
                logger.error(f"Found NaN value in {key}.")
                logger.info("list_w : %s", list_w)
                continue  # 跳过包含 NaN 值的键
            pbar_encrypted_list = tqdm(list_w, position=3, leave=False)
            for n in pbar_encrypted_list:
                encrypted_list.append(pub_key.encrypt(n))
            end_time = time.time()
            inter_time = end_time - start_time
            formated_time = time_formate(inter_time)
            logger.info('encrypting key [%s].length : %d, cost time : %s', key, len(list_w), formated_time)
            encrypted_state_dict[key] = encrypted_list
        else:
            list_w = state_dict[key].view(-1).cpu().tolist()
            inter_time = time.time() - start_time
            formated_time = time_formate(inter_time)
            encrypted_state_dict[key] = list_w
            logger.info('load key [%s].length : %d, cost time : %s', key, len(list_w), formated_time)

    return encrypted_state_dict


def model_decrypt(encrypted_model_state_dict, private_key, model_shape_type, keys_to_encrypt):
    """
    this method is to decrypt the model with Server's private_key
    :param encrypted_model_state_dict: {'key', list}
    :param private_key: Server's private_key
    :param model_shape_type: model's shape and dtype
    :param keys_to_encrypt: List of keys of model to decrypt
    """
    decrypted_state_dict = {}  # store the decrypted model parameters
    start_time = time.time()
    for key, value in encrypted_model_state_dict.items():
        if key in keys_to_encrypt:
            decrypted_list = [private_key.decrypt(ciphertext) for ciphertext in value]  # decrypted the value
        else:
            decrypted_list = value
        original_dtype = model_shape_type[key]['dtype']
        original_shape = model_shape_type[key]['shape']
        tensor_param = torch.tensor(decrypted_list, dtype=original_dtype).reshape(original_shape)
        decrypted_state_dict[key] = tensor_param
        logger.info("decrypting key [%s].shape [%s]", key, decrypted_state_dict[key].shape)
    inter_time = time.time() - start_time
    formated_time = time_formate(inter_time)
    logger.info("decrypt time : [%s]", formated_time)
    return decrypted_state_dict


def save_client_weight(n_data, client_dict, select_clients_list, server):
    """
    this method is to set client weight
    :param n_data: data selected by server from clients
    :param client_dict: client state dict
    :param select_clients_list: list of selected clients
    :param server: server
    """
    logger.info("server.n_data : %d", n_data)
    for client_id in select_clients_list:
        client_dict[client_id].set_client_weight(float(client_dict[client_id].n_data) / float(n_data))
        server.client_weight[client_id] = client_dict[client_id].weight
        logger.info("client_dict[%s].weight : %f", client_id, client_dict[client_id].weight)


def cal_and_set_secret_number(client_dict, select_clients):
    """
    this method is clients cooperate with each other to recover secret number(not the secret number generated by server).
    :param client_dict: client state dict
    """
    secret_number = 0.0
    for client_id in select_clients:
        secret_number += client_dict[client_id].weight * client_dict[client_id].share
    for client_id in select_clients:
        secret_number_64=np.array(secret_number).astype(np.float64)
        client_dict[client_id].set_secret_number(secret_number_64)
    logger.info("recover secret number by clients: %f", secret_number_64)


def generate_secret_number(seed):
    """
    system generates secret number
    :param seed: input the random seed
    :return: secret number
    """
    bit_length = 12 # max
    random.seed(seed)
    max_value = (1 << bit_length) - 1
    random_integer = random.randint(0, max_value)
    return random_integer


def generate_and_split_secret_number(client_dict, seed):
    """
    this method is to generate and split secret number into n parts.
    :param client_dict: client state dict.
    :param seed: input the random seed
    """
    logger.info('generating secret number and splitting it ...')
    n = len(client_dict)
    if n <= 0:
        raise ValueError("the number of client need larger than 0.")
    secret_number = generate_secret_number(seed)
    logger.info('secret_number : %f', secret_number)
    random.seed(seed)
    parts = [random.randint(0, secret_number) for _ in range(n - 1)]
    last_part = secret_number - sum(parts)
    while last_part <= 0:
        parts = [random.randint(0, secret_number) for _ in range(n - 1)]
        last_part = secret_number - sum(parts)
    parts.append(last_part)
    random.shuffle(parts)
    for i, client_id in enumerate(client_dict):
        part_64 = np.array(parts[i]).astype(np.float64)
        client_dict[client_id].set_share(part_64)
        logger.info('client_dict[%s].share : %f', client_id, client_dict[client_id].share)


def test_accuracy_of_global_model(global_model, test_set):
    """
    System tests the model on test dataset.
    :param global_model: global model
    :param test_set: test dataset
    :return: global model's accuracy of this global round.
    """
    test_loader = DataLoader(test_set, batch_size=200, shuffle=True)
    gpu = 0
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    global_model.to(device)
    accuracy_collector = 0
    for step, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            b_x = x.to(device)  # Tensor on GPU
            b_y = y.to(device)  # Tensor on GPU

            test_output = global_model(b_x)
            pred_y = torch.max(test_output, 1)[1].to(device).data.squeeze()
            accuracy_collector = accuracy_collector + sum(pred_y == b_y)
    accuracy = accuracy_collector / len(test_set)
    return accuracy.cpu().numpy()

def time_formate(inter_time):
    """
    formate the cost time.
    :param inter_time: inter time
    :return: formatted time
    """
    minutes, seconds = divmod(inter_time, 60)
    seconds, milliseconds = divmod(seconds, 1)
    # 格式化时间为 "m分钟 s秒 ms毫秒" 形式
    formated_time = f"{int(minutes)}m:{int(seconds)}s:{int(milliseconds * 1000)}ms"
    return formated_time