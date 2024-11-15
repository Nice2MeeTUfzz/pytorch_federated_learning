import re


def parse_time(time):
    """
    解析时间字符串，返回总毫秒数。
    :param time: 时间字符串，格式为 '0m:1s:29ms'
    :return: 总毫秒数
    """
    parts = time.split(':')
    minutes = int(parts[0].strip('m'))
    seconds = int(parts[1].strip('s'))
    milliseconds = int(parts[2].strip('ms'))
    total_ms = minutes * 60 * 1000 + seconds * 1000 + milliseconds
    return total_ms


def format_time(total_ms):
    """
    将总毫秒数转换回 '0m:1s:29ms' 格式。
    :param total_ms: 总毫秒数
    :return: 时间字符串
    """
    minutes, remainder = divmod(total_ms, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes}m:{seconds}s:{milliseconds}ms"


def local_model_train_cost_time():
    pattern = r'local model train cost time : (\d+m:\d+s:\d+ms)'
    cost_time = []
    with open('./results/MNIST/Homo_LeNet_5_20/result.log', 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                time_str = match.group(1)
                cost_time.append(time_str)
                # print(f"Extracted time: {time_str}")

    list = [parse_time(t) for t in cost_time]
    average_ms = sum(list) / len(list)
    print(average_ms)


def client_encrypt_cost_time():
    client_times = {}
    with open('./results/MNIST/Homo_LeNet_5_20/result.log', 'r') as file:
        for line in file:
            if 'local model encrypting' in line:
                current_client_id = line.split('client_dict[')[1].split(']')[0]
                client_times[current_client_id] = []
            elif 'encrypting key' in line:
                time_str = line.split('cost time : ')[1].strip()
                client_times[current_client_id].append(time_str)

    # 计算每个客户端的总加密时间和平均加密时间
    total_times = {}
    total_time = 0
    for client_id, times in client_times.items():
        total_ms = sum(parse_time(t) for t in times)
        total_times[client_id] = total_ms
        total_time += total_ms
    print(total_times)
    average_time = format_time(total_time / len(client_times))
    print(average_time)

def agg_cost_time():
    agg_time = []
    with open('./results/MNIST/Homo_LeNet_5_20/result.log', 'r') as file:
        for line in file:
            if 'Server aggregate model done, cost time' in line:
                time_str = line.split('cost time : ')[1].strip()
                agg_time.append(time_str)
    print(agg_time)
    total_ms = sum(parse_time(t) for t in agg_time)
    average = format_time(total_ms/len(agg_time))
    print(average)

def server_decrypt_cost_time():
    decrypt_time = []
    with open('./results/MNIST/Homo_LeNet_5_20/result.log', 'r') as file:
        for line in file:
            if 'decrypt time :' in line:
                # 提取解密时间
                time_str = line.split('decrypt time : [')[1].strip().strip(']')
                decrypt_time.append(time_str)
    print(decrypt_time)
    total_ms = sum(parse_time(t) for t in decrypt_time)
    average = format_time(total_ms/len(decrypt_time))
    print(average)
if __name__ == '__main__':
    server_decrypt_cost_time()