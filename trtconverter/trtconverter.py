"""
This module:
1. downloads model from HuggingFace Hub;
2. converts it to ONNX
3. optimizes it with TensorRT
4. saves TensorRT plan to disk
"""
import logging
import os
import sys

import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import tensorrt as trt
from PIL import Image


MODEL_NAME = "segformer-b4"
MODEL_HUGGINGFACE_NAME = "nvidia/segformer-b4-finetuned-ade-512-512"
INPUT_SHAPE = (1, 3, 512, 512)
OUTPUT_PATH = "out/"
ENABLE_FP16 = True


def load_model():
    logging.info(f"loading model from {MODEL_HUGGINGFACE_NAME}")

    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_HUGGINGFACE_NAME)
    preprocessor = SegformerImageProcessor.from_pretrained(MODEL_HUGGINGFACE_NAME)
    return model, preprocessor

def save_preprocessor_config(preprocessor, out_path):
    logging.info(f"saving preprocessor config to {out_path}")
    
    os.makedirs(out_path, exist_ok=True)
    preprocessor.save_pretrained(out_path)

def preprocess_inputs(img, preprocessor):
    logging.info("preprocessing inputs")

    inputs = preprocessor(images=img, return_tensors="pt")
    return inputs

def load_sample_image(img_path, preprocessor):
    logging.info(f"loading sample: {img_path}")

    img = Image.open(img_path)
    img = preprocess_inputs(img, preprocessor)
    return img

def _get_cuda_device():
    # Check for GPU
    if not torch.cuda.is_available():
        return None
    device = torch.device('cuda')

    return device

def _export_to_onnx(device, 
                    pytorch_model, 
                    input_shape, 
                    onnx_model_path, 
                    model_name,
                    opset_ver=17):
    logging.info(f"converting model {model_name} to onnx to {onnx_model_path}")
    os.makedirs(onnx_model_path, exist_ok=True)
    onnx_model_path = os.path.join(onnx_model_path, f"{model_name}_opset{opset_ver}.onnx")

    example_input = torch.randn(input_shape).to(device)
    pytorch_model.to(device)

    torch.onnx.export(pytorch_model,
                      example_input,
                      onnx_model_path,
                      export_params=True,
                      verbose=True,
                      opset_version=opset_ver)
    
    return onnx_model_path


def _build_engine(onnx_model_path,
                  input_shape,
                  fp16=False,
                  max_gpu_mem=1 << 33,  # 2^33 = 1GB
                  trt_model_path=None
                  ):
    logging.info("building tensorrt engine")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(1) as network, \
            builder.create_builder_config() as config, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # allow inference on newer TRT engines
        config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
       
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_gpu_mem)
       
        # read ONNX
        with open(onnx_model_path, 'rb') as model:
            parser.parse(model.read())
        
        # build engine
        network.get_input(0).shape = input_shape
        ser_engine = builder.build_serialized_network(network, config)

        if ser_engine is None:
            logging.error("tensorrt couldn't build trt engine")
            sys.exit()
       
        logging.info("saving trt engine")

        # generate path and filename for output model
        onnx_path, onnx_filename = os.path.split(onnx_model_path)
        trt_filename = onnx_filename.rsplit('.', 1)[0]  # use onnx filename as a base
        if fp16:
            trt_filename += "_fp16"
        trt_prime_ver = trt.__version__.split('.')[0]
        trt_filename += f"_trt{trt_prime_ver}.trt"
        trt_path = trt_model_path if trt_model_path is not None else onnx_path
        os.makedirs(trt_path, exist_ok=True)
        trt_filepath = os.path.join(trt_path, trt_filename)

        # save TensorRT plan
        with open(trt_filepath, 'wb') as f:
            f.write(ser_engine)


def convert_to_trt(device, 
                   pytorch_model, 
                   input_shape, 
                   out_path, 
                   model_name="model"):
    # conversion pipeline
    onnx_path = _export_to_onnx(device, pytorch_model, input_shape, out_path, model_name)
    _build_engine(onnx_path, input_shape, fp16=ENABLE_FP16)


def main():
    logging.basicConfig(level=logging.INFO)

    device = _get_cuda_device()
    if device is None:
        logging.error("gpu/cuda is not available. This script requires nvidia gpu to work.")
        return

    pytorch_model, preprocessor = load_model()
    save_preprocessor_config(preprocessor, OUTPUT_PATH)

    convert_to_trt(device, pytorch_model, INPUT_SHAPE, OUTPUT_PATH, MODEL_NAME)


if __name__ == "__main__":
    main()
