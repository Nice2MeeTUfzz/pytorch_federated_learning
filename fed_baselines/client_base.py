from copy import deepcopy
from importlib.metadata import metadata
from time import sleep
import time
import logging
from utils.models import *
from torch.utils.data import DataLoader
from utils.fed_utils import assign_dataset, init_model, gaussian_noise, time_formate
from tqdm import tqdm

torch.set_default_dtype(torch.float64)
logger = logging.getLogger('client_base')
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('result.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class FedClient(object):
    def __init__(self, name, epoch, dataset_id, model_name, batch_size, add_noise, lr, i_seed):
        """
        Initialize the client k for federated learning.
        :param name: Name of the client k
        :param epoch: Number of local training epochs in the client k
        :param dataset_id: Local dataset in the client k
        :param model_name: Local model in the client k
        """
        # Initialize the metadata in the local client
        self.target_ip = '127.0.0.3'
        self.port = 9999
        self.name = name

        # Initialize the parameters in the local client
        self._epoch = epoch
        self._batch_size = batch_size
        self._lr = lr
        self._momentum = 0.5
        self.num_workers = 4
        self.loss_rec = []
        self.n_data = 0
        self.weight = 0.0  # client weight each global round

        # Initialize the local training and testing dataset
        self.trainset = None

        # Initialize the local model
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class,
                                image_channel=self._image_channel)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.param_len = sum([np.prod(p.size()) for p in model_parameters])

        # Training on GPU
        gpu = 0
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

        # gaussian_noise
        self.add_noise = add_noise
        self.sigma = 1
        self.clip = 0.1
        self.seed = i_seed

        # model encrypt parameters
        self.public_key = None
        self.share = 0.0
        self.secret_number = 0.0
        self.global_epoch = 0

    def load_trainset(self, trainset):
        """
        Client loads the training dataset.
        :param trainset: Dataset for training.
        """
        self.trainset = trainset
        self.n_data = len(trainset)

    def set_client_weight(self, weight):
        """
        Client sets the weight each global round
        :param weight: Weight
        """
        self.weight = weight

    def set_secret_number(self, secret_number):
        """
        Client sets the secret number each global round
        :param secret_number: Secret number
        """
        self.secret_number = secret_number

    def set_share(self, share):
        """
        Client sets share.
        :param share: Share distributed by Secret number
        """
        self.share = share

    def set_global_epoch(self, global_epoch):
        """
        Client sets the global epoch each global round.
        :param global_epoch: Global epoch
        """
        self.global_epoch = global_epoch

    def set_public_key(self, public_key):
        """
        Client sets the public key each global round.
        :param public_key: Server's public key.
        """
        self.public_key = public_key

    def update(self, model_state_dict):
        """
        Client updates the model from the server.
        :param model_state_dict: Global model.
        """
        self.model = init_model(model_name=self.model_name, num_class=self._num_class,
                                image_channel=self._image_channel)
        self.model.load_state_dict(model_state_dict)

    def recover_model(self):
        start_time = time.time()
        state_dict = self.model.state_dict()
        for key in state_dict:
            # if self.model.state_dict()[key].dtype == torch.int64:
            #     self.model.state_dict()[key] = self.model.state_dict()[key].to(torch.float64)
            secret_tensor = torch.tensor(self.secret_number, dtype=torch.float64, device=state_dict[key].device)
            state_dict[key] -= secret_tensor
        inter_time = time.time() - start_time
        formated_time = time_formate(inter_time)
        logger.info("recover model cost time : %s", formated_time)
        self.model.load_state_dict(state_dict)

    def train(self):
        """
        Client trains the model on local dataset
        :return: Local updated model, number of local data points, training loss
        """
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True,
                                  num_workers=self.num_workers)

        self.model.to(self._device)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        loss_func = nn.CrossEntropyLoss()

        # Training process
        pbar_client_train = tqdm(range(self._epoch), position=2, leave=False)  # 设置进度条
        train_start_time = time.time()
        for epoch in pbar_client_train:
            logger.info("client_dict[%s] training epoch : %d", self.name, epoch)
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    optimizer.zero_grad()
                    loss.backward()
                    """
                    是否加入DP噪声，在yaml文件中设置
                    """
                    if self.add_noise:
                        for param in self.model.parameters():
                            if param.grad is not None:
                                print("before_add:", param.grad.data)
                                param.grad.data += gaussian_noise(param.grad.data.shape, self.clip, self.sigma,
                                                                  generator=self.seed,
                                                                  device=self._device)
                                print("after_add", param.grad.data)

                    optimizer.step()

                pbar_client_train.set_description(
                    'Client Epoch %d' % epoch)
        train_inter_time = time.time() - train_start_time
        train_formated_time = time_formate(train_inter_time)
        logger.info("local model train cost time : %s", train_formated_time)
        """
        保存训练后的参数到.pth文件,可不保存
        """
        # torch.save(self.model.state_dict(), 'model_state_mnist.pth')

        # model encode
        start_time = time.time()
        for key in self.model.state_dict():
            secret_tensor = torch.tensor(self.share, dtype=torch.float64, device=self._device)
            if torch.isnan(secret_tensor).any() or torch.isinf(secret_tensor).any():
                logger.error("secret_tensor for %s contains NaN or Inf values: %s", key, secret_tensor)
                continue
            self.model.state_dict()[key] += secret_tensor
        inter_time = time.time() - start_time
        formated_time = time_formate(inter_time)
        logger.info("model encode time : %s", formated_time)
        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()
        # return encrypted_state_dict, self.n_data, loss.data.cpu().numpy()
