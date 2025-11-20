import torch

print("CUDA Available: ", torch.cuda.is_available())

if torch.cuda.is_available():
	print("Device count: ", torch.cuda.device_count())
	print("Current device index: ", torch.cuda.current_device())
	print("Current device name: ", torch.cuda.get_device_name(0))
else:
	print("CUDA is not available. check driver / Pytorch install.")
