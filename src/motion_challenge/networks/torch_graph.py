"""Safe tensor weights and dynamic-batch execution of feed-forward ONNX graphs."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from motion_challenge.validation import sha256_file

OPS = {"Identity", "Concat", "Sub", "Div", "Add", "Mul", "MatMul", "Gemm",
       "LayerNormalization", "Mish", "Relu", "Elu", "Sigmoid", "Tanh", "Softplus", "Erf"}


class TensorGraph(torch.nn.Module):
    def __init__(self, directory, device="cpu", expected_source=None):
        super().__init__()
        directory = Path(directory)
        self.graph = json.loads((directory / "graph.json").read_text())
        if expected_source and self.graph["source_sha256"] != expected_source:
            raise ValueError("Converted graph source identity differs")
        if sha256_file(directory / "weights.safetensors") != self.graph["weights_sha256"]:
            raise ValueError("Converted weights checksum differs")
        self.keys = {}
        for i, (name, value) in enumerate(load_file(str(directory / "weights.safetensors"), device=device).items()):
            key = f"tensor_{i}"
            self.register_buffer(key, value)
            self.keys[name] = key

    def forward(self, **inputs):
        if set(inputs) != set(self.graph["inputs"]):
            raise ValueError("Graph input names differ")
        values = {name: getattr(self, key) for name, key in self.keys.items()}
        values.update(inputs)
        for node in self.graph["nodes"]:
            x = [values[k] for k in node["inputs"]]
            a, op = node["attributes"], node["op"]
            if op == "Identity": y = x[0]
            elif op == "Concat": y = torch.cat(x, dim=a["axis"])
            elif op == "Add": y = x[0] + x[1]
            elif op == "Sub": y = x[0] - x[1]
            elif op == "Mul": y = x[0] * x[1]
            elif op == "Div": y = x[0] / x[1]
            elif op == "MatMul": y = x[0] @ x[1]
            elif op == "Gemm":
                left = x[0].T if a.get("transA", 0) else x[0]
                right = x[1].T if a.get("transB", 0) else x[1]
                y = a.get("alpha", 1.) * (left @ right)
                if len(x) > 2: y = y + a.get("beta", 1.) * x[2]
            elif op == "LayerNormalization":
                axis = a.get("axis", -1)
                y = F.layer_norm(x[0], x[0].shape[axis:], x[1], x[2] if len(x) > 2 else None, a.get("epsilon", 1e-5))
            elif op == "Mish": y = F.mish(x[0])
            elif op == "Relu": y = F.relu(x[0])
            elif op == "Elu": y = F.elu(x[0], alpha=a.get("alpha", 1.))
            elif op == "Sigmoid": y = x[0].sigmoid()
            elif op == "Tanh": y = x[0].tanh()
            elif op == "Erf": y = x[0].erf()
            elif op == "Softplus": y = F.softplus(x[0], threshold=float("inf"))
            else: raise ValueError(f"Unsupported operator: {op}")
            values[node["output"]] = y
        return {name: values[name] for name in self.graph["outputs"]}


def convert(source, directory, expected_sha256, outputs=None):
    import onnx
    from onnx import helper, numpy_helper
    source, directory = Path(source), Path(directory)
    if sha256_file(source) != expected_sha256:
        raise ValueError("Source checkpoint checksum differs")
    model = onnx.load(source)
    outputs = outputs or [v.name for v in model.graph.output]
    producers = {o: n for n in model.graph.node for o in n.output}
    required = set()
    def visit(name):
        if name in required: return
        required.add(name)
        if name in producers:
            for arg in producers[name].input: visit(arg)
    for name in outputs:
        if name not in {v.name for v in model.graph.output}: raise ValueError("Unknown output")
        visit(name)
    inputs = {v.name: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
              for v in model.graph.input if v.name in required}
    ranks = {len(shape) for shape in inputs.values()}
    if ranks not in ({1}, {2}): raise ValueError("Feed-forward converter requires rank-one or rank-two inputs")
    lifted = ranks == {1}
    weights = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy())
               for t in model.graph.initializer if t.name in required}
    nodes = []
    for node in model.graph.node:
        if not any(o in required for o in node.output): continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "Constant":
            if set(attrs) != {"value"}: raise ValueError("Unsupported constant encoding")
            weights[node.output[0]] = torch.from_numpy(numpy_helper.to_array(attrs["value"]).copy())
            continue
        if node.op_type not in OPS or len(node.output) != 1:
            raise ValueError(f"Unsupported operator: {node.op_type}")
        if lifted and node.op_type in {"Concat", "LayerNormalization"} and attrs.get("axis", -1) >= 0:
            attrs["axis"] += 1
        if node.op_type == "Gemm" and attrs.get("transA", 0):
            raise ValueError("Transposed batch input requires model-specific conversion")
        nodes.append({"op": node.op_type, "inputs": list(node.input), "output": node.output[0], "attributes": attrs})
    directory.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(directory / "weights.safetensors"))
    graph = {"source_sha256": expected_sha256, "weights_sha256": sha256_file(directory / "weights.safetensors"),
             "inputs": inputs, "outputs": outputs, "lifted_batch_axis": lifted, "nodes": nodes,
             "converter_sha256": sha256_file(__file__)}
    (directory / "graph.json").write_text(json.dumps(graph, indent=2)+"\n")
    return graph


def validate(source, directory, devices=("cpu", "cuda:0"), batches=(1, 7, 40, 256), atol=2e-5, rtol=2e-5):
    import numpy as np
    from onnx.reference import ReferenceEvaluator
    reference = ReferenceEvaluator(str(source))
    graph = json.loads((Path(directory) / "graph.json").read_text())
    rng = np.random.default_rng(1847)
    maximum = 0.
    cases = []
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    for batch in batches:
        feed = {}
        for name, shape in graph["inputs"].items():
            width = shape[-1]
            feed[name] = rng.normal(size=(batch, width)).astype(np.float32)
        oracle = {k: [] for k in graph["outputs"]}
        for i in range(batch):
            single = {k: v[i] if graph["lifted_batch_axis"] else v[i:i+1] for k, v in feed.items()}
            for key, value in zip(graph["outputs"], reference.run(graph["outputs"], single)):
                oracle[key].append(value.reshape(-1))
        expected = {k: np.stack(v) for k, v in oracle.items()}
        for device in devices:
            model = TensorGraph(directory, device)
            with torch.inference_mode():
                actual = model(**{k: torch.tensor(v, device=device) for k, v in feed.items()})
            errors = {}
            for key, value in actual.items():
                value = value.cpu().numpy()
                np.testing.assert_allclose(value, expected[key], rtol=rtol, atol=atol)
                errors[key] = float(np.max(np.abs(value-expected[key])))
                maximum = max(maximum, errors[key])
            cases.append({"batch": batch, "device": device, "max_abs_error": errors})
    report = {"status": "passed", "reference": "original ONNX ReferenceEvaluator per sample",
              "tolerance":{"absolute":atol,"relative":rtol},"source_sha256": sha256_file(source), "graph_sha256": sha256_file(Path(directory)/"graph.json"),
              "weights_sha256": graph["weights_sha256"], "max_abs_error": maximum, "cases": cases}
    (Path(directory)/"parity.json").write_text(json.dumps(report, indent=2)+"\n")
    return report
