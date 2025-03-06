import torch
import options.options as option
from models import create_model

from tqdm import tqdm
import torch.nn.functional as F

def auto_padding(img, times=16):
    """
    이미지 크기를 `times`의 배수로 맞추기 위해 패딩 적용 (PyTorch 버전)

    Args:
        img (torch.Tensor): (B, C, H, W) 형태의 텐서
        times (int): 맞추고 싶은 배수 (기본값=16)

    Returns:
        padded_img (torch.Tensor): 패딩이 적용된 이미지
        padding (list): [h1, h2, w1, w2] (위, 아래, 왼쪽, 오른쪽 패딩 크기)
    """
    _, _, H, W = img.shape

    # H와 W를 `times`의 배수로 맞추기 위한 패딩 크기 계산
    h_pad = (times - H % times) % times
    w_pad = (times - W % times) % times

    # 위/아래, 왼쪽/오른쪽 패딩 분배
    h1, h2 = h_pad // 2, h_pad - h_pad // 2
    w1, w2 = w_pad // 2, w_pad - w_pad // 2

    # 패딩 적용 (REFLECT 모드 사용)
    padded_img = F.pad(img, (w1, w2, h1, h2), mode='reflect')

    return padded_img

def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)
    return model

device = torch.device('cuda:0')
conf_path = "./code/confs/LOL.yml"
model = load_model(conf_path)

model.netG = model.netG.to(device)
model.net_hq = model.net_hq.to(device)

# Dummy input
# 입력 크기 정의 (예: 720x1080 이미지, 채널 3)
# input_size = (720, 1080)
# input_size = (832, 658) # Exdark
input_size = (900, 1600) # nuImages

dummy_input = torch.randn(1, 3, *input_size).cuda()
dummy_input = auto_padding(dummy_input, times=16)

# GPU warm-up (10회 실행)
for _ in range(10):
    _ = model.get_sr(lq=dummy_input, heat=None)

# 100번의 추론 실행 후 시간 측정
times = []
for _ in tqdm(range(100)):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    _ = model.get_sr(lq=dummy_input, heat=None)
    end_event.record()

    torch.cuda.synchronize()
    times.append(start_event.elapsed_time(end_event))

# 평균 실행 시간 계산 및 출력
avg_time = sum(times) / len(times)
print("Average inference time over 100 runs: {:.3f} ms".format(avg_time))