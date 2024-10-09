from utils.models import *
from copy import deepcopy
import time
from tqdm import tqdm


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


def model_encrypt(ori_model, pub_key, keys_to_encrypt):
    """
    this method is to encrypt the model with Paillier encryption, use pub_key and choose several keys to encrypt.
    :param ori_model: Model name
    :param pub_key: Paillier public key
    :param keys_to_encrypt: List of keys of model to encrypt
    :return: encrypted model, encrypted_model_state_dict[key]=(encrypted_list, meta_data{'shape','dtype'})
    """
    encrypted_model = deepcopy(ori_model)
    encrypted_state_dict = {}
    state_dict = encrypted_model.state_dict()
    num_keys_to_test = 1
    counter = 0
    start_time = time.time()
    for key in state_dict.keys():
        if counter >= num_keys_to_test:
            break
        if key in keys_to_encrypt:
            list_w = state_dict[key].view(-1).cpu().tolist()
            pbar_encrypted_list = tqdm(list_w, position=3, leave=False)
            encrypted_list = []
            for n in pbar_encrypted_list:
                encrypted_list.append(pub_key.encrypt(n))
            inter_time = time.time() - start_time
            pbar_encrypted_list.set_description(f'encrypting the {key},cost time: {inter_time}')
            encrypted_state_dict[key] = encrypted_list
        else:
            encrypted_state_dict[key] = state_dict[key]
        counter += 1
    return encrypted_state_dict


def model_decrypt(encrypted_model, private_key, model_shape_type):
    """
    this method is to decrypt the model with Server's private_key
    :param encrypted_model: {'key', list}
    :param private_key: Server's private_key
    :param model_shape_type: model's shape and dtype
    """
    decrypted_state_dict = {} # store the decrypted model parameters
    for key, encrypted_list in encrypted_model.items():
        decrypted_list = [private_key.decrypt(ciphertext) for ciphertext in encrypted_list]# decrypted the value
        original_dtype = model_shape_type[key].dtype
        original_shape = model_shape_type[key].shape
        tensor_param = torch.tensor(decrypted_list, dtype=original_dtype).reshape(original_shape)
        decrypted_state_dict[key] = tensor_param
    return decrypted_state_dict

def save_client_weight(n_data,client_dict, select_clients_list):
    """
    this method is to set client weight
    :param n_data: data selected by server from clients
    :param client_dict: client state dict
    :param select_clients_list: list of selected clients
    """
    for client_id in select_clients_list:
        client_dict[client_id].set_weight(client_dict[client_id].n_data/n_data)

def cal_secret_number():
