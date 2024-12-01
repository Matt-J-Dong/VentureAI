# import bitsandbytes as bnb
# print(bnb.__version__)
# print(bnb.has_cuda)

import bitsandbytes as bnb

#print(bnb.__version__)  # Ensure the version is printed
try:
    print(bnb.cuda.is_available())  # Check CUDA availability
except AttributeError:
    print("CUDA check method not found in this version.")
