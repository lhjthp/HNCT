# HNCT

**About PyTorch 1.1.0**

- There have been minor changes with the 1.1.0 update. Now we support PyTorch 1.1.0 by default, and please use the legacy branch if you prefer older version.
  We provide scripts for reproducing all the results from our paper. You can train your model from scratch, or use a pre-trained model to enlarge your images.

## Dependencies

- Python 3.6
- PyTorch >= 1.0.0
- numpy
- skimage
- **imageio**
- matplotlib
- tqdm
- cv2 >= 3.xx (Only if you want to use video input/output)
- openpyxl
- pandas

## Code

Clone this repository into any place you want.

````bash
git clone https://github.com/lhjthp/HNCT.git

## Quickstart (Demo)
Run the script in ``src`` folder. Before you run the demo, please uncomment the appropriate line in ```demo.sh``` that you want to execute.
```bash
cd src       # You are now in */HNCT/src
sh demo.sh
````

You can find the result images from `experiment/test/` folder.

| Model | Scale | File name (.pt) | Parameters | **PSNR/SSIM** (Set5 Set14 BSD100 Urban100 Manga109)              |
| ----- | ----- | --------------- | ---------- | ---------------------------------------------------------------- |
| ---   | 2     | x2.pt           | 356K       | 38.08/0.9608 33.65/0.9182 32.22/0.9001 32.22/0.9294 38.87/0.9774 |
| ---   | 3     | x3.pt           | 363K       | 34.47/0.9275 30.44/0.8439 29.15/0.8067 28.28/0.8557 33.81/0.9459 |
| ---   | 4     | x4.pt           | 372K       | 32.31/0.8957 28.71/0.7834 27.63/0.7381 26.20/0.7896 30.70/0.9112 |

You can evaluate your models with widely-used benchmark datasets:

## How to test HNCT

We used [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset to train our model.

Unpack the tar file to any place you want. Then, change the `dir_data` argument in `src/option.py` to the place where DIV2K images are located.

We recommend you to pre-process the images before training. This step will decode all **png** files and save them as binaries. Use `--ext sep_reset` argument on your first run. You can skip the decoding part and use saved binaries with `--ext sep` argument.

You can train HNCT by yourself. All scripts are provided in the `src/demo.sh`.

```bash
cd src       # You are now in */HNCT/src
sh demo.sh
```

**Update log**
