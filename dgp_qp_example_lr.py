import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import OrderedDict

import torch
import torchvision.utils as vutils

import utils
from models import DGP_qp_lr

sys.path.append("./")

# Arguments for demo
def add_example_parser(parser):
    parser.add_argument(
        '--image_path', type=str, default='',
        help='Path of the image to be processed (default: %(default)s)')
    parser.add_argument(
        '--class', type=int, default=-1,
        help='class index of the image (default: %(default)s)')
    parser.add_argument(
        '--image_path2', type=str, default='',
        help='Path of the 2nd image to be processed, used in "morphing" mode (default: %(default)s)')
    parser.add_argument(
        '--class2', type=int, default=-1,
        help='class index of the 2nd image, used in "morphing" mode (default: %(default)s)')

    parser.add_argument(
        '--k_lr', type=float, default=1,
        help='learning rate ratios (sensitivity amplification) (default: %(default)s)')
    
    parser.add_argument('--n_qubits', type=int, default=20, help='number of qubits')
    parser.add_argument('--n_qlayers', type=int, default=3, help='number of layers')
    parser.add_argument('--n_heads', type=int, default=6, help='number of heads')
    parser.add_argument('--measurement_setting', type=str, default='d', help='quantum mode')
    return parser


# prepare arguments and save in config
parser = utils.prepare_parser()
parser = utils.add_dgp_parser(parser)
parser = add_example_parser(parser)
config = vars(parser.parse_args())
utils.dgp_update_config(config)

# set random seed
utils.seed_rng(config['seed'])

source_dir = os.path.dirname(os.path.abspath(__file__))
config['exp_path'] = source_dir

config['image_path'] = source_dir + config['image_path']


if not os.path.exists('{}/images'.format(config['exp_path'])):
    os.makedirs('{}/images'.format(config['exp_path']))
if not os.path.exists('{}/images_sheet'.format(config['exp_path'])):
    os.makedirs('{}/images_sheet'.format(config['exp_path']))


# 讨论trainable prior 在图片修复方面的能力, 主要是inplainting
dgp_qp_lr = DGP_qp_lr(config)

# prepare the target image
img = utils.get_img(config['image_path'], config['resolution']).cuda()
category = torch.Tensor([config['class']]).long().cuda()
dgp_qp_lr.set_target(img, category, config['image_path'])


loss_dict = dgp_qp_lr.run()

########################################################
if config['dgp_mode'] == 'category_transfer':
    save_imgs = img.clone().cpu()
    inputs = torch.rand((1, config['n_heads'] * config['n_qubits'])) * 2 - 1
    inputs = inputs.cuda()
    for i in range(151, 294):  # dog & cat
        # for i in range(7, 25):  # bird
        i_label = torch.Tensor([i]).long().cuda()
        with torch.no_grad():
            x = dgp_qp_lr.model(inputs, i_label, 1.0)
            utils.save_img(
                x[0],
                '%s/images/%s_class%d.jpg' % (config['exp_path'], dgp_qp_lr.img_name, i))
            save_imgs = torch.cat((save_imgs, x.cpu()), dim=0)
    vutils.save_image(
        save_imgs,
        '%s/images_sheet/%s_categories.jpg' % (config['exp_path'], dgp_qp_lr.img_name),
        nrow=int(save_imgs.size(0) ** 0.5),
        normalize=True)
############# morphing 不做了
elif config['dgp_mode'] == 'morphing':   # 比较耗时，因为需要另外新建2个模型。
    dgp_qp_2 = DGP_qp_lr(config)
    dgp_qp_interp = DGP_qp_lr(config)

    img2 = utils.get_img(config['image_path2'], config['resolution']).cuda()
    category2 = torch.Tensor([config['class2']]).long().cuda()

    dgp_qp_2.set_target(img2, category2, config['image_path2'])
    # dgp_qp_2.select_z(select_y=True if config['class2'] < 0 else False)
    loss_dict = dgp_qp_2.run()  # better with a different seed

    weight1 = dgp_qp_lr.model.state_dict()  # 是不是可以全部当模型插入使用
    weight2 = dgp_qp_2.model.state_dict()
    weight_interp = OrderedDict()
    save_imgs = []
    inputs = torch.zeros(1, config['n_heads'] * config['n_qubits']).cuda()
    with torch.no_grad():
        for i in range(11):
            alpha = i / 10
            # interpolate between both latent vector and generator weight
            # z_interp = alpha * dgp_qp.z + (1 - alpha) * dgp_qp_2.z
            # 实际上是embedding
            y_interp = alpha * dgp_qp_lr.model.G.shared(dgp_qp_lr.y) + (1 - alpha) * dgp_qp_2.model.G.shared(dgp_qp_2.y)
            for k, w1 in weight1.items():
                w2 = weight2[k]
                weight_interp[k] = alpha * w1 + (1 - alpha) * w2
            dgp_qp_interp.model.load_state_dict(weight_interp)  # 所有的参数都要load，而不只是G
            # 不训练，所以采样，此时让温度为0即可。
            x_interp = dgp_qp_interp.model.forward(inputs, y_interp, temperature=1e-8)  # 参数已经插值了，输入还是0
            save_imgs.append(x_interp.cpu())
            # save images
            save_path = '%s/images/%s_%s' % (config['exp_path'], dgp_qp_lr.img_name, dgp_qp_2.img_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            utils.save_img(x_interp[0], '%s/%03d.jpg' % (save_path, i + 1))
        save_imgs = torch.cat(save_imgs, 0)
    vutils.save_image(
        save_imgs,
        '%s/images_sheet/morphing_%s_%s.jpg' % (config['exp_path'], dgp_qp_lr.img_name, dgp_qp_2.img_name),
        nrow=int(save_imgs.size(0) ** 0.5),
        normalize=True)
