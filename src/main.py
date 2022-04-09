import torch
import numpy as np
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['CUDA_CACHE_PATH']='~/.cudacache'

# torch.manual_seed(args.seed)
torch.cuda.empty_cache()
checkpoint = utility.checkpoint(args)
# python /home/linhanjiang/projects/AIM/EDSR/src/main.py --model rfdn --data_test Set5+Set14+
# B100+Urban100+DIV2K  --data_range 801-900 --scale 4 --save rfdn_x4 --pre_train /home/linhanjiang/projects/AIM/EDSR/experiment/test/model/model_best.pt --rgb_range 1 --test_only --save_results
def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
