import os
import copy
import numpy as np
import click
from typing import List, Optional
import torch
import pickle
import dnnlib
import legacy

def extract_conv_names(model, model_res):
    model_names = list(name for name,weight in model.named_parameters())

    return model_names

def blend_models(low, high, model_res, resolution):

    resolutions =  [4*2**x for x in range(int(np.log2(resolution)-1))]
    # print(resolutions)
    
    low_names = extract_conv_names(low, model_res)
    high_names = extract_conv_names(high, model_res)

    assert all((x == y for x, y in zip(low_names, high_names)))

    #start with lower model and add weights above
    model_out = copy.deepcopy(low)
    params_src = high.named_parameters()
    dict_dest = model_out.state_dict()

    for name, param in params_src:
        if not any(f'synthesis.b{res}' in name for res in resolutions) and not ('mapping' in name):
            # print(name)
            dict_dest[name].data.copy_(param.data)

    model_out_dict = model_out.state_dict()
    model_out_dict.update(dict_dest) 
    model_out.load_state_dict(dict_dest)
    
    return model_out

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--lower_res_pkl', help='Network pickle filename for lower resolutions', required=True)
@click.option('--higher_res_pkl', help='Network pickle filename for higher resolutions', required=True)
@click.option('--output_path','out', help='Network pickle filepath for output', default='./blended.pkl')
@click.option('--model_res', type=int, help='Output resolution of model (likely 1024, 512, or 256)', default=1024, show_default=True)
@click.option('--split_res', 'resolution', type=int, help='Resolution to split model weights', default=64, show_default=True)

def create_blended_model(
    ctx: click.Context,
    lower_res_pkl: str,
    higher_res_pkl: str,
    model_res: Optional[int],
    resolution: Optional[int],
    out: Optional[str],
):

	G_kwargs = dnnlib.EasyDict()

	with dnnlib.util.open_url(lower_res_pkl) as f:
	    lo = legacy.load_network_pkl(f, custom=False, **G_kwargs) # type: ignore
	    lo_G, lo_D, lo_G_ema = lo['G'], lo['D'], lo['G_ema']

	with dnnlib.util.open_url(higher_res_pkl) as f:
	    hi = legacy.load_network_pkl(f, custom=False, **G_kwargs)['G_ema'] # type: ignore

	model_out = blend_models(lo_G_ema, hi, model_res, resolution)
	# for n in model_out.named_parameters():
	#     print(n[0])

	data = dict([('G', None), ('D', None), ('G_ema', None)])
	with open(out, 'wb') as f:
	    data['G'] = lo_G
	    data['D'] = lo_D
	    data['G_ema'] = model_out
	    pickle.dump(data, f)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    create_blended_model() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------