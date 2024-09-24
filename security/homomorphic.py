import copy
import time
from tqdm import tqdm
import torch
from utils.models import *
from phe import paillier
import struct
from gmpy2 import mpz, powmod, f_div, invert, is_prime, random_state, mpz_urandomb, rint_round, log2, gcd


def float_to_ieee754(value):
    # 将浮点数转换为二进制字符串
    binary_str = format(struct.unpack('!I', struct.pack('!f', value))[0], '032b')
    ieee754_int = int(binary_str, 2)
    return ieee754_int


def ieee754_to_float(ieee754_str):
    # 检查输入是否为32位的二进制字符串
    if len(ieee754_str) != 32:
        raise ValueError("Input must be a 32-bit binary string")

    value_int = int(ieee754_str, 2)
    packed = struct.pack('!I', value_int)
    # 将字节串转换为浮点数
    float_num = struct.unpack('!f', packed)[0]
    return float_num


def test_model_parameters():
    global_pub_key, global_priv_key = paillier.generate_paillier_keypair()
    pub_key = global_pub_key
    priv_key = global_priv_key
    model_path = './model_state_mnist.pth'

    checkpoint = torch.load(model_path)

    model = LeNet(num_classes=10, in_channels=1)

    model.load_state_dict(checkpoint)

    w = model.state_dict()
    print('encrypting...')
    time_start = time.time()
    for k in w.keys():
        # print(f"k:{k}")
        list_w = w[k].view(-1).cpu().tolist()
        pbar = tqdm(list_w)
        for i, elem in enumerate(pbar):
            '''
            将elemt由float浮点数转换为IEEE 754 格式二进制字符串
            再将字符串转为大整数
            '''
            elem_int = float_to_ieee754(elem)
            # elem_float = format(elem_int, '032b') # 这个用不到
            list_w[i] = pub_key.encrypt(elem_int)
            # list_w[i] = pub_key.encrypt(elem)
            pbar.set_description(f"-----{k}-----")
        w[k] = list_w
    time_end = time.time()
    print(f"Encryption time:{time_end - time_start}")

    print(w)
    torch.save(w, './model_encrypt_para.pth')


def test_homomorphic_with_float(pk):
    global list_new
    model_path_1 = './model_state_mnist.pth'
    model_path_2 = './model_state_mnist.pth'

    checkpoint_1 = torch.load(model_path_1)
    checkpoint_2 = torch.load(model_path_2)

    model1 = LeNet(num_classes=10, in_channels=1)
    model2 = LeNet(num_classes=10, in_channels=1)

    model1.load_state_dict(checkpoint_1)
    model2.load_state_dict(checkpoint_2)

    w1 = model1.state_dict()
    w2 = model2.state_dict()

    update_list = {}

    print('encrypting...')
    time_start = time.time()
    for k1, k2 in zip(w1.keys(), w2.keys()):
        list_w1 = w1[k1].view(-1).cpu().tolist()
        list_w2 = w2[k2].view(-1).cpu().tolist()
        update_list[k1] = []
        pbar = tqdm(enumerate(zip(list_w1, list_w2)), total=len(list_w1))
        for i, (v1, v2) in pbar:
            #         '''
            #         将elemt由float浮点数转换为IEEE 754 格式二进制字符串
            #         再将字符串转为大整数
            #         '''
            #         elem_int = float_to_ieee754(elem)
            #         # elem_float = format(elem_int, '032b') # 这个用不到
            # list_w1[i] = pub_key.encrypt(v1)
            # list_w2[i] = pub_key.encrypt(v2)
            # print(f"v1+v2={v1+v2}")
            enc_v1 = pk.encrypt(v1)
            enc_v2 = pk.encrypt(v2)
            # print(f"i:{list_w1[i]}")
            # print(f"k1:{k1}")
            # print(f"k2:{k2}")
            update_list[k1].append(enc_v1 + enc_v2)
            # print(priv_key.decrypt(list_new[i]) == priv_key.decrypt(list_w1[i])+priv_key.decrypt(list_w2[i]))
            pbar.set_description(f"-----{k1}-----")
            break
        # origin_shape = list(model_new.state_dict()[k1].size())
        # model_new[k1] = torch.FloatTensor(w3[k1]).to('cpu').view(*origin_shape)
        # w3[k1] = list_new
    print(update_list)
    time_end = time.time()
    time_ = time_end - time_start
    print(time_)
    return update_list
    # torch.save(model_new, './model_encrypt_para.pth')
    # return model_new

    # pass


def decrypt_model_parameters(w,sk):
    update = copy.deepcopy(w)
    print("decrypting.......")
    time_s = time.time()
    for k in update:
        print(f"update {k}={update[k]}")
        dec_v = sk.decrypt(update[k][0])
        print(dec_v)
    pass


if __name__ == '__main__':
    # test_model_parameters()
    global_pub_key, global_priv_key = paillier.generate_paillier_keypair()  # 生成公私钥对
    pub_key = global_pub_key
    priv_key = global_priv_key
    en_model_par = test_homomorphic_with_float(pub_key)
    decrypt_model_parameters(en_model_par,priv_key)
    # model_path_encrypt = './model_encrypt_para.pth'
    # checkpoint_encrypt = torch.load(model_path_encrypt)
    # model_encrypt = LeNet(num_classes=10, in_channels=1)
    # model_encrypt.load_state_dict(checkpoint_encrypt)
    # w1 = model_encrypt.state_dict()
    # print(w1)
