import os
import numpy as np
import torch
import click
import PIL.Image
import dnnlib
import legacy

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--npzs', help='comma separated .npz files', type=str, required=True, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')

def combine_npz(
    ctx: click.Context,
    npzs: str,
    outdir: str,
):
	print('Combining .npz files...')
	files = npzs.split(',')

	os.makedirs(outdir, exist_ok=True)

	ws = torch.tensor(())
	for i,f in enumerate(files):
		print(f)
		w = torch.tensor(np.load(f)['w'])
		ws = torch.cat((ws,w), 0)

	print(ws.size())
	np.savez(f'{outdir}/combined.npz', w=ws.numpy())


#----------------------------------------------------------------------------

if __name__ == "__main__":
    combine_npz() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------