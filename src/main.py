import torch
import numpy as np
# 导入 u 文件
import utility
import data
import model
import loss
#导入参数
from option import args
from trainer import Trainer
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_CACHE_PATH'] = '~/.cudacache'

# torch.manual_seed(args.seed)
torch.cuda.empty_cache()
# 构建检查点变量，调用
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
        # 完成初始化
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            # Trainer 至少构造了一个对象，两个方法
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            # 完成（关闭日志）
            checkpoint.done()


if __name__ == '__main__':
    main()
