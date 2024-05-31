from model.detector import Detector
import utils
import torch
from dataset import Dataset, collate_fn
from model.evaluator import CocoDetectionEvaluator

device = torch.device("cpu")
configuration = utils.load_configuration("config.yaml")
batch_size = configuration["batch_size"]
num_workers = configuration["num_workers"]

train_dataset = Dataset(configuration, augment=True)
test_dataset = Dataset(configuration, test=True)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True,
    collate_fn=collate_fn,
    pin_memory=True,
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    collate_fn=collate_fn,
    pin_memory=True,
)
evaluator = CocoDetectionEvaluator(configuration["classes"], device)

checkpoint = "checkpoints/2024-05-29_13-36-42/40_0.22216349751641154"

model = Detector(len(configuration["classes"]), False).to(device)
# model.load_state_dict(torch.load(f"{checkpoint}/weights.pt"))

model.eval()
qconfig = torch.ao.quantization.QConfig(
    activation=torch.ao.quantization.observer.HistogramObserver.with_args(
        reduce_range=True
    ),
    weight=torch.ao.quantization.observer.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric
    ),
)

# qconfig = torch.ao.quantization.QConfig(
#     activation=torch.ao.quantization.observer.HistogramObserver.with_args(
#         reduce_range=True
#     ),
#     weight=torch.ao.quantization.observer.MinMaxObserver.with_args(
#         dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
#     ),
# )

qconfig = torch.ao.quantization.get_default_qconfig("x86")


model.qconfig = qconfig
# model_fused = model.fuse_modules()
model_prepared = torch.ao.quantization.prepare(model)
train_dataloader.augment = False
evaluator.compute_map(test_dataloader, model_prepared)
model_int8 = torch.ao.quantization.convert(model_prepared)
map05 = evaluator.compute_map(test_dataloader, model_int8)
print(f"mAP05: {map05}")
train_dataloader.augment = True
