TF_CUDNN_USE_AUTOTUNE=0

import sys
from PIL import Image

sys.path.insert(1, 'pose_tensorflow')

from util.config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input
cfg = {}
cfg['cfg'] = load_config("pose_tensorflow/demo/pose_cfg.yaml")
cfg['sess'], cfg['inputs'], cfg['outputs'] = predict.setup_pose_prediction(cfg['cfg'])

def resize_image(img: Image):
    basewidth = 300

    wpercent = basewidth / img.size[0]
    hsize = int(img.size[1] * wpercent)
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)

    return img


def get_pose(image, d=cfg):
    image = resize_image(image)

    image_batch = data_to_input(image)

    outputs_np = d['sess'].run(d['outputs'], feed_dict={d['inputs']: image_batch})
    scmap, locref, _ = predict.extract_cnn_output(outputs_np, d['cfg'])

    pose = predict.argmax_pose_predict(scmap, locref, d['cfg'].stride)
    return pose
