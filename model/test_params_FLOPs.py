from model.arch import DyConvSR_v2, dyconvsr_v2
import config
from modules.model_summary import *
import config

from torchsummary import summary


if __name__ == "__main__":
    model = dyconvsr_v2(config)
    
    # count number of parameters
    num_params = get_model_parameters_number(model)
    print(f"Params: {num_params}")
    
    # count number of flops
    flops_count = get_model_flops(model, input_res=(3, 1080, 720), print_per_layer_stat=False, input_constructor=None)
    print(f"Flops: {flops_count}")
