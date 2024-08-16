// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3
const ModelProto = $root.onnx.ModelProto;
// const VERSION = 19;
// const IR_VERSION = 8; // $root.onnx.Version.IR_VERSION;
const IN1K_MEAN = [0.485, 0.456, 0.406];
const IN1K_SD = [0.229, 0.224, 0.225];

const AttributeType = $root.onnx.AttributeProto.AttributeType;
const TensorType = $root.onnx.TensorProto.DataType;

const ATTR_TYPE_MAP = {
  FLOAT: { value: AttributeType.FLOAT, key: "f", name: "FLOAT" },
  INT: { value: AttributeType.INT, key: "i", name: "INT" },
  STRING: { value: AttributeType.STRING, key: "s", name: "STRING" },
  TENSOR: { value: AttributeType.TENSOR, key: "t", name: "TENSOR" },
  GRAPH: { value: AttributeType.GRAPH, key: "g", name: "GRAPH" },
  FLOATS: { value: AttributeType.FLOATS, key: "floats", name: "FLOATS" },
  INTS: { value: AttributeType.INTS, key: "ints", name: "INTS" },
  STRINGS: { value: AttributeType.STRINGS, key: "strings", name: "STRINGS" },
  TENSORS: { value: AttributeType.TENSORS, key: "tensors", name: "TENSORS" },
  GRAPHS: { value: AttributeType.GRAPHS, key: "graphs", name: "GRAPHS" },
};

const INOUT_TENSOR_TYPE_MAP = {
  "tensor(bool)": TensorType.BOOL,
  "tensor(uint8)": TensorType.UINT8,
  "tensor(int8)": TensorType.INT8,
  "tensor(uint16)": TensorType.UINT16,
  "tensor(int16)": TensorType.INT16,
  "tensor(uint32)": TensorType.UINT32,
  "tensor(int32)": TensorType.INT32,
  "tensor(float)": TensorType.FLOAT,
  "tensor(uint64)": TensorType.UINT64,
  "tensor(int64)": TensorType.INT64,
  "tensor(double)": TensorType.DOUBLE,
  "tensor(bfloat16)": null,
  "tensor(float16)": null,
  "tensor(complex128)": null,
  "tensor(complex64)": null,
  "tensor(string)": null,
  // https://arxiv.org/abs/2209.05433
  "tensor(float8e5m2)": null,
  "tensor(float8e4m3fn)": null,
  // graphcore, amd, qualcomm
  "tensor(float8e5m2fnuz)": null,
  "tensor(float8e4m3fnuz)": null,
};

const TENSOR_TYPE_MAP = {
  [TensorType.BOOL]: Uint8Array,
  [TensorType.UINT8]: Uint8Array,
  [TensorType.INT8]: Int8Array,
  [TensorType.UINT16]: Uint16Array,
  [TensorType.INT16]: Int16Array,
  [TensorType.UINT32]: Uint32Array,
  [TensorType.INT32]: Int32Array,
  [TensorType.FLOAT]: Float32Array,
  [TensorType.DOUBLE]: Float64Array,
  [TensorType.UINT64]: BigUint64Array,
  [TensorType.INT64]: BigInt64Array,
  [TensorType.UNDEFINED]: () => {
    throw new Error("Can't support UNDEFINED tensor");
  },
  [TensorType.STRING]: () => {
    throw new Error("Can't support STRING");
  },
  [TensorType.FLOAT16]: () => {
    throw new Error("Can't support FLOAT16");
  },
  [TensorType.BFLOAT16]: () => {
    throw new Error("Can't support BFLOAT16");
  },
  [TensorType.COMPLEX64]: () => {
    throw new Error("Can't support COMPLEX64");
  },
  [TensorType.COMPLEX128]: () => {
    throw new Error("Can't support COMPLEX128");
  },
  [TensorType.FLOAT8E4M3FN]: () => {
    throw new Error("Can't support FLOAT8E4M3FN");
  },
  [TensorType.FLOAT8E4M3FNUZ]: () => {
    throw new Error("Can't support FLOAT8E4M3FNUZ");
  },
  [TensorType.FLOAT8E5M2]: () => {
    throw new Error("Can't support FLOAT8E5M2");
  },
  [TensorType.FLOAT8E5M2FNUZ]: () => {
    throw new Error("Can't support FLOAT8E5M2FNUZ");
  },
};
const TENSOR_TYPE_INV = Object.fromEntries(
  Object.entries(TensorType).map(([k, v]) => [v, k])
);
