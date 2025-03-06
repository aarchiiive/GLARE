import torch
import options.options as option
from models import create_model

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
# dummy_input = torch.randn(1, 3, 720, 1080).to(device)
dummy_input = torch.randn(1, 3, 900, 1600).to(device)

# GPU warm-up (10회 실행)
for _ in range(10):
    _ = model.get_sr(lq=dummy_input, heat=None)

# 100번의 추론 실행 후 시간 측정
times = []
for _ in range(20):
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