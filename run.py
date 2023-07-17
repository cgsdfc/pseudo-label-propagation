# MIT License

# Copyright (c) 2023 Ao Li, Cong Feng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from src.models.pseudolabel import train_main
from pathlib import Path as P
import logging
import warnings
from src.vis.visualize import *
from src.utils.io_utils import *
import matplotlib.pyplot as plt

warnings.filterwarnings(action="ignore")
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    method = 'plp'
    dataname = 'Digits-10.mat'
    datapath = P("./data/").joinpath(dataname)
    eta = 50
    savedir = P(f"output/{method}/{dataname}-{eta}")

    train_main(
        datapath=datapath,
        eta=eta / 100,
        device='cpu',
        savedir=savedir,
        save_vars=True,
        pp_type='G',
        lamda=0.1,
    )

    H_common = load_var(savedir, "H_common")
    sns.heatmap(pairwise_distances(H_common))
    plt.show()
    plot_scatters(H_common, datapath=datapath, palette='hls')
    plt.show()
