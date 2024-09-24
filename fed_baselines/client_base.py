from copy import deepcopy
from time import sleep
import time
from security.secure_utils import global_pub_key
from utils.models import *
from torch.utils.data import DataLoader
from utils.fed_utils import assign_dataset, init_model, gaussian_noise
from tqdm import tqdm


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
        self._momentum = 0.9
        self.num_workers = 4
        self.loss_rec = []
        self.n_data = 0

        # Initialize the local training and testing dataset
        self.trainset = None
        self.test_data = None

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

        # model encryption
        # self.pk = pub_key

    def load_trainset(self, trainset):
        """
        Client loads the training dataset.
        :param trainset: Dataset for training.
        """
        self.trainset = trainset
        self.n_data = len(trainset)

    def update(self, model_state_dict):
        """
        Client updates the model from the server.
        :param model_state_dict: Global model.
        """
        self.model = init_model(model_name=self.model_name, num_class=self._num_class,
                                image_channel=self._image_channel)
        self.model.load_state_dict(model_state_dict)

    def train(self):
        """
        Client trains the model on local dataset
        :return: Local updated model, number of local data points, training loss
        """
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True,
                                  num_workers=self.num_workers)

        self.model.to(self._device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        loss_func = nn.CrossEntropyLoss()

        # Training process
        pbar_client_train = tqdm(range(self._epoch), position=2, leave=False)   # 设置进度条

        for epoch in pbar_client_train:
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
        """
        保存训练后的参数到.pth文件,可不保存
        """
        # torch.save(self.model.state_dict(), 'model_state_mnist.pth')
        """
        return 分割后的模型参数
        """
        # 模型加密
        encrypted_model = deepcopy(self.model)
        encrypted_state_dict = {}
        state_dict = encrypted_model.state_dict()
        start_time = time.time()
        for k in state_dict.keys():
            list_w = state_dict[k].view(-1).cpu().tolist()
            pbar_encrypted_list = tqdm(list_w, position=3, leave=False)
            encrypted_list = []
            for n in pbar_encrypted_list:
                encrypted_list.append(global_pub_key.encrypt(n))
            inter_time = time.time() - start_time
            pbar_encrypted_list.set_description(f'encrypting the {k},cost time: {inter_time}')
            encrypted_state_dict[k] = encrypted_list
        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()