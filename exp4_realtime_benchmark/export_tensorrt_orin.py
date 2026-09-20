import argparse
from pathlib import Path

import tensorrt as trt
import torch

from benchmark_realtime_pc import WEIGHTS, load_state
from models import CNN, DiffusionPolicy, FlowPolicy


ROOT = Path(__file__).resolve().parent
ENGINES = ROOT / "engines"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_file, engine_file, fp16):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(Path(onnx_file).read_bytes()):
        errors = "\n".join(str(parser.get_error(i))
                           for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errors}")
    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(f"TensorRT build failed: {onnx_file}")
    Path(engine_file).write_bytes(engine)
    print(f"saved: {engine_file}")


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model.encode(images)


class DPStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, action, timestep, features):
        return self.model.forward_cached(action, timestep, features)


class GVCStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, action, timestep):
        return self.model.forward_cached(features, action, timestep)


def export(model, inputs, name, fp16):
    model.eval().cuda()
    args = ((torch.randn(inputs, device="cuda"),)
            if all(isinstance(x, int) for x in inputs) else inputs)
    onnx_file = ENGINES / f"{name}.onnx"
    torch.onnx.export(
        model, args, onnx_file, opset_version=17,
        input_names=[f"input_{i}" for i in range(len(args))],
        output_names=["output"], dynamic_axes=None)
    print(f"onnx: {onnx_file}")
    build_engine(onnx_file, ENGINES / f"{name}.engine", fp16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    ENGINES.mkdir(exist_ok=True)
    device, fp16 = torch.device("cuda"), args.precision == "fp16"

    cnn = load_state(CNN(), WEIGHTS / "cnn_cassini.pth", device)
    export(cnn, (1, 3, 224, 224), "cnn", fp16)

    dp = load_state(DiffusionPolicy(),
                    WEIGHTS / "diffusion_policy_cassini.pth", device)
    export(Encoder(dp), (1, 1, 3, 224, 224), "dp_encoder", fp16)
    export(DPStep(dp), (
        torch.randn(1, 2, 2, device=device),
        torch.zeros(1, dtype=torch.long, device=device),
        torch.randn(1, 512, device=device)), "dp_step", fp16)

    gvc = load_state(FlowPolicy(),
                     WEIGHTS / "flow_matching_cassini.pth", device)
    export(Encoder(gvc), (1, 1, 3, 224, 224), "gvc_encoder", fp16)
    export(GVCStep(gvc), (
        torch.randn(1, 24, device=device),
        torch.randn(1, 4, device=device),
        torch.zeros(1, 1, device=device)), "gvc_step", fp16)


if __name__ == "__main__":
    main()
