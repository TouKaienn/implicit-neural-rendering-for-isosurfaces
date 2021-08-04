import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--save_txt_root_path', type=str,
                    default='E:/VScodelib/FCNet/data/input_data')
parser.add_argument('--input_image', type=str, default='E:/VScodelib/FCNet/data/tiny_vorts0008_normalize_dataset')
parser.add_argument('--output_shape', type=int, default=512)
parser.add_argument('--batch_size', type = int, default=1)
config = parser.parse_args()