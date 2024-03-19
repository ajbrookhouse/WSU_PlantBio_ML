import torch
check=torch.cuda.is_available()
if check==True:
    print("YES. You are using GPU")
else:
    print("WARNING! GPU is NOT in use")
