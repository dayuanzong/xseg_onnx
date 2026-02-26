import os
import sys
from pathlib import Path
import argparse
import numpy as np
import cv2
import collections

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.leras import nn
from facelib import XSegNet
import tf2onnx
from tf2onnx import tf_loader
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto


def _find_xseg_output():
    graph = nn.tf.get_default_graph()
    sig_ops = [op for op in graph.get_operations() if op.type == "Sigmoid"]
    sig_ops = [op for op in sig_ops if "XSeg" in op.name] or sig_ops
    if not sig_ops:
        raise Exception("未找到 XSeg Sigmoid 输出")
    return sig_ops[-1].outputs[0]


def _build_model(resolution, model_root, cpu):
    if cpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    try:
        nn.initialize_main_env()
    except Exception:
        pass
    device_config = nn.DeviceConfig.CPU()
    if not cpu:
        try:
            device_config = nn.DeviceConfig.GPUIndexes([0])
        except Exception:
            device_config = nn.DeviceConfig.CPU()
    nn.initialize(device_config=device_config)
    xseg = XSegNet(
        name="XSeg",
        resolution=resolution,
        load_weights=True,
        weights_file_root=model_root,
        raise_on_no_model_files=True,
        place_model_on_cpu=cpu,
        run_on_cpu=cpu,
    )
    nn.tf_sess.run(nn.tf.global_variables_initializer())
    try:
        for model, filename in xseg.model_filename_list:
            model.load_weights(model_root / filename)
    except Exception:
        pass
    try:
        uninit = nn.tf_sess.run(nn.tf.report_uninitialized_variables())
        if len(uninit) > 0:
            uninit_names = set([n.decode("utf-8") for n in uninit])
            vars_to_init = [v for v in nn.tf.global_variables() if v.name.split(":")[0] in uninit_names]
            if vars_to_init:
                nn.tf_sess.run(nn.tf.variables_initializer(vars_to_init))
    except Exception:
        pass
    pred = _find_xseg_output()
    return xseg, pred


def _export_onnx(xseg, pred, output_path, opset):
    input_names = [xseg.input_t.name]
    output_names = [pred.name]
    graph_def = tf_loader.freeze_session(nn.tf_sess, input_names=input_names, output_names=output_names)
    tf2onnx.convert.from_graph_def(
        graph_def,
        input_names=input_names,
        output_names=output_names,
        opset=opset,
        output_path=str(output_path),
    )


def _fix_convtranspose_asymmetric_pads(onnx_path):
    model = onnx.load(str(onnx_path))
    graph = model.graph
    existing_names = {n.name for n in graph.node}
    existing_names.update(i.name for i in graph.initializer)

    def _unique(name):
        if name not in existing_names:
            existing_names.add(name)
            return name
        idx = 1
        while f"{name}_{idx}" in existing_names:
            idx += 1
        new_name = f"{name}_{idx}"
        existing_names.add(new_name)
        return new_name

    out_to_consumers = collections.defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            out_to_consumers[inp].append(node)

    new_nodes = []
    for node in graph.node:
        if node.op_type != "ConvTranspose":
            continue
        pads_attr = None
        pads = None
        for attr in node.attribute:
            if attr.name == "pads":
                pads_attr = attr
                pads = list(attr.ints)
                break
        if pads != [0, 0, 1, 1]:
            continue
        pads_attr.ints[:] = [0, 0, 0, 0]
        slice_out = _unique(f"{node.output[0]}_slice")
        for consumer in out_to_consumers.get(node.output[0], []):
            for i, inp in enumerate(consumer.input):
                if inp == node.output[0]:
                    consumer.input[i] = slice_out
        starts_name = _unique(f"{slice_out}_starts")
        ends_name = _unique(f"{slice_out}_ends")
        axes_name = _unique(f"{slice_out}_axes")
        steps_name = _unique(f"{slice_out}_steps")
        graph.initializer.extend([
            helper.make_tensor(starts_name, TensorProto.INT64, [2], [0, 0]),
            helper.make_tensor(ends_name, TensorProto.INT64, [2], [-1, -1]),
            helper.make_tensor(axes_name, TensorProto.INT64, [2], [2, 3]),
            helper.make_tensor(steps_name, TensorProto.INT64, [2], [1, 1]),
        ])
        slice_node = helper.make_node(
            "Slice",
            [node.output[0], starts_name, ends_name, axes_name, steps_name],
            [slice_out],
            name=_unique(f"{node.name}_slice"),
        )
        new_nodes.append(slice_node)

    if new_nodes:
        graph.node.extend(new_nodes)
        onnx.save(model, str(onnx_path))


def _prepare_input(resolution, image_path=None):
    if image_path:
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("无法读取图片")
        img = img.astype(np.float32) / 255.0
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
    else:
        rng = np.random.default_rng(42)
        img = rng.random((resolution, resolution, 3), dtype=np.float32)
    return img


def _postprocess_mask(mask):
    mask = np.clip(mask, 0, 1.0)
    mask[mask < 0.1] = 0
    return mask


def _test_consistency(xseg, pred, onnx_path, resolution, image_path=None):
    img = _prepare_input(resolution, image_path)
    ref = xseg.extract(img)
    ref = _postprocess_mask(ref)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: img[None, ...]})[0]
    if out.ndim == 4 and out.shape[1] == 1:
        out = out[:, 0, :, :]
    if out.ndim == 3:
        out = out[0]
    out = _postprocess_mask(out)

    diff = np.abs(ref - out)
    return float(diff.max()), float(diff.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--test-image", type=str, default=None)
    args = parser.parse_args()

    default_model_root = Path(__file__).resolve().parents[3] / "model_generic_xseg"
    default_output_path = default_model_root / f"XSeg_{args.resolution}.onnx"
    model_root = Path(args.model_root) if args.model_root else default_model_root
    output_path = Path(args.output) if args.output else default_output_path

    print("请输入模型目录路径（包含 XSeg_*.npy）")
    model_root_input = input(f"模型目录路径（默认：{model_root}）：").strip()
    if model_root_input:
        model_root = Path(model_root_input)
    print("请输入输出 ONNX 路径")
    output_input = input(f"输出路径（默认：{output_path}）：").strip()
    if output_input:
        output_path = Path(output_input)
    print("请输入用于一致性测试的切脸图片路径（可留空）")
    test_image_input = input(f"切脸图片路径（默认：{args.test_image or ''}）：").strip()
    if test_image_input:
        args.test_image = test_image_input

    xseg, pred = _build_model(args.resolution, model_root, args.cpu)
    _export_onnx(xseg, pred, output_path, args.opset)
    _fix_convtranspose_asymmetric_pads(output_path)

    max_diff, mean_diff = _test_consistency(xseg, pred, output_path, args.resolution, args.test_image)
    print(f"ONNX 导出完成: {output_path}")
    print(f"一致性检查 - max_diff: {max_diff:.8f}, mean_diff: {mean_diff:.8f}")


if __name__ == "__main__":
    main()
