import torch
check=torch.cuda.is_available()
if check==True:
    print("True. You are using GPU")
else:
    print("False. WARNING! GPU is NOT in use")
