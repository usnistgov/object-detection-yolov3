

def convert(pt_model_filepath, onnx_model_filepath):
    import torch
    import numpy as np

    model = torch.load(pt_model_filepath)
    dummy_input = np.random.rand(1, 3, 224, 224)
    dummy_input = torch.cuda.FloatTensor(dummy_input)
    torch.onnx.export(model, dummy_input, onnx_model_filepath, verbose=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert a pytorch (pt) model file format to ONNC format.')
    parser.add_argument('--pt_model_filepath', type=str, required=True, help='Filepath to the .pt pytorch model file to be converted to onnx.')
    parser.add_argument('--onnx_model_filepath', type=str, required=True, help='Filepath to where the *onnx model file should be saved.')
    args = parser.parse_args()

    convert(args.pt_model_filepath, args.onnx_model_filepath)