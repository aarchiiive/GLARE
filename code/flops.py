import torch

from fvcore.nn import FlopCountAnalysis

import options.options as option
from models import create_model
from utils.util import opt_get

import warnings
warnings.filterwarnings("ignore")


def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    # model_path = opt_get(opt, ['model_path'], None)
    # model.load_network(load_path=model_path, network=model.netG)
    return model, opt

device = torch.device('cuda:0')
# conf_path = "./code/confs/train_stage2_LOL.yml"
conf_path = "./code/confs/LOL.yml"
model, opt = load_model(conf_path)

model.eval()
# model.half()

with torch.no_grad():
    model.netG = model.netG.to(device)
    model.net_hq = model.net_hq.to(device)

    # Dummy input
    dummy_input = torch.randn(1, 3, 720, 1080).to(device)
    # dummy_input = torch.randn(1, 3, 360, 540).to(device)
    # dummy_input = torch.randn(1, 3, 900, 1600).to(device)

    flops = FlopCountAnalysis(model, dummy_input)

# 모델 파라미터 수 계산
param_count = sum(p.numel() for p in model.netG.parameters())

# GFLOP 계산 (1 GFLOP = 10^9 FLOP)
gflops = flops.total() / 1e9

# 결과 출력
print(f"FLOPs: {flops.total()}")  # 총 FLOP 수 출력
print(f"Parameters: {param_count}")  # 총 파라미터 수 출력
print(f"GFLOPs: {gflops:.6f}")  # GFLOPs 출력