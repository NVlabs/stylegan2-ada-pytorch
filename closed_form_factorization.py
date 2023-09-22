import argparse
import torch
import dnnlib
import legacy
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract factor/eigenvectors of latent spaces using closed form factorization"
    )
    parser.add_argument(
        "--out", type=str, default="factor.pt", help="name of the result factor file"
    )
    parser.add_argument("--ckpt", type=str, help="name of the model checkpoint")
    args = parser.parse_args()

    custom = False

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.size = None 
    G_kwargs.scale_type = 'pad'
    
    print('Loading networks from "%s"...' % args.ckpt)
    device = torch.device('cuda')
    with dnnlib.util.open_url(args.ckpt) as f:
        G = legacy.load_network_pkl(f, custom=custom, **G_kwargs)['G_ema'].to(device) # type: ignore


    # device = torch.device('cuda')
    # with dnnlib.util.open_url(args.ckpt) as f:
    #     G = pickle.load(f)['G_ema'].to(device) # type: ignore

    modulate = {
        k[0]: k[1]
        for k in G.named_parameters()
        if "affine" in k[0] and "torgb" not in k[0] and "weight" in k[0] or ("torgb" in k[0] and "b4" in k[0] and "weight" in k[0] and "affine" in k[0])
    }

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V.to("cpu")

    torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.out)
