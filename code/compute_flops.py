import torch
import torchprofile

import options.options as option
from models import create_model
# from utils.util import opt_get

import warnings
warnings.filterwarnings("ignore")

def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)
    print(opt)
    return model, opt

# Device 설정
device = torch.device('cuda:0')
conf_path = "./code/confs/LOL.yml"
model, opt = load_model(conf_path)

model.eval()

# 모델을 GPU로 이동
model.netG = model.netG.to(device)
model.net_hq = model.net_hq.to(device)

# 더미 입력 생성
dummy_input = torch.randn(1, 3, 720, 1080).to(device) # darkface
dummy_input = torch.randn(1, 3, 720, 1080).to(device)

# torchprofile을 사용하여 FLOPs 측정
flops = torchprofile.profile_macs(model, dummy_input)

# 모델 파라미터 수 계산
param_count = sum(p.numel() for p in model.netG.parameters())

# GFLOPs 계산 (1 GFLOP = 10^9 FLOP)
gflops = flops / 1e9

# 결과 출력
print(f"FLOPs: {flops:.0f}")  # 총 FLOP 수 출력
print(f"Parameters: {param_count}")  # 총 파라미터 수 출력
print(f"GFLOPs: {gflops:.6f}")  # GFLOPs 출력