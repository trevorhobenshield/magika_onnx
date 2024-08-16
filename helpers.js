function argmax(preds, labels) {
  const idx = preds.reduce((iMax, x, i, arr) => (x > arr[iMax] ? i : iMax), 0);
  return preds[idx] >= labels[idx].threshold ? labels[idx].name : null;
}

function bufferToTensor(buf, dim = 1536) {
  const ui8 = new Uint8Array(buf);
  const ui8Padded = new Uint8Array(dim);
  ui8Padded.set(ui8.slice(0, dim));
  const f32 = new Float32Array(ui8Padded);
  return new ort.Tensor("float32", f32, [1, dim]);
}

function parseNpy(buffer) {
  const dtype = {
    "|u1": Uint8Array,
    "<i1": Int8Array,
    "|i1": Int8Array,
    "<u2": Uint16Array,
    "<i2": Int16Array,
    "<u4": Uint32Array,
    "<i4": Int32Array,
    "<f4": Float32Array,
    "<f8": Float64Array,
  };

  const magicLength = 6;
  const majorOffset = 6;
  const minorOffset = 7;
  const offsetLength = 8;

  const u8 = new Uint8Array(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 6));
  if (magic !== "\x93NUMPY") {
    throw new Error(`Invalid .npy format: magic = ${magic}`);
  }

  const major = u8[majorOffset];
  const minor = u8[minorOffset];
  const lenLength = major >= 2 ? 4 : 2;
  const view = new DataView(buffer);

  let headerLen;
  if (major === 1 && minor === 0) {
    headerLen = view.getUint16(offsetLength, true);
  } else if ([2, 3].includes(major) && minor === 0) {
    headerLen = view.getUint32(offsetLength, true);
  } else {
    throw new Error(`Unsupported .npy version: ${major}.${minor}`);
  }

  const decoder = new TextDecoder(
    major === 3 && minor === 0 ? "utf-8" : "ascii"
  );
  const headerText = decoder.decode(
    u8.subarray(offsetLength + lenLength, offsetLength + lenLength + headerLen)
  );
  const headerJson = headerText
    .toLowerCase()
    .replace(/'/g, '"')
    .replace("(", "[")
    .replace(/,*\),*/g, "]");
  const { descr, shape, fortran_order } = JSON.parse(headerJson);

  if (fortran_order) {
    throw new Error("Fortran-ordered arrays not supported");
  }

  const start = magicLength + 2 + lenLength + headerLen;
  const data = new dtype[descr](buffer, start);

  return {
    major,
    minor,
    descr,
    dims: shape,
    data,
  };
}

/**
 * // Usage
 * const f32 = TensorType.FLOAT;
 * const shape = [2, 5];
 * const tensor = RandomTensor[f32](2, 5);
 * @type {{[p: string]: function(...[*]): *}}
 */
const RandomTensor = Object.fromEntries(
  Object.entries(TENSOR_TYPE_MAP).map(([dtype, arrayType]) => [
    dtype,
    (...shape) => {
      dtype = Number(dtype);
      const size = shape.reduce((a, b) => a * b, 1);
      const isFloat = dtype === TensorType.FLOAT || dtype === TensorType.DOUBLE;
      const isBool = dtype === TensorType.BOOL;
      let counter = 1;
      return new ort.Tensor(
        arrayType.from({ length: size }, () => {
          if (isFloat) {
            return Math.random() * 2 - 1;
          } else if (isBool) {
            return Math.random() > 0.5 ? 1 : 0;
          } else {
            return counter++;
          }
        }),
        shape
      );
    },
  ])
);

function getInOut(model) {
  function fn(k) {
    return Object.fromEntries(
      model.graph[k].map(({ name, type }) => [
        name,
        {
          type: TENSOR_TYPE_INV[type.tensorType.elemType],
          dims: type.tensorType.shape.dim.map((y) => y.dimValue),
        },
      ])
    );
  }

  return [fn("input"), fn("output")];
}

function getInfo(file, model) {
  console.log(`%c${file.name}`, "color: #db2777");
  console.log("      IR Version:", model.irVersion);
  console.log("   Opset Version:", model.opsetImport[0].version);
  console.log("        Producer:", model.producerName);
  console.log("Producer Version:", model.producerVersion);
  console.log("      Graph Name:", model.graph.name);
  [modelOps, unsupportedOps] = checkOps(model);
  console.log(`       Operators:`, modelOps);
  const [ins, outs] = getInOut(model);
  fmtInOut(ins, outs);
  console.log(`         Backend: ${SESSION_OPTIONS.executionProviders}`);
  console.log("model =", model);
  console.log("session =", session);
}

function getShape(arr) {
  let res = [];
  let tmp = arr;
  while (Array.isArray(tmp)) {
    res.push(tmp.length);
    tmp = tmp[0];
  }
  return res;
}

function _zip(arr, ...arrs) {
  return arr.map((x, i) => [x, ...arrs.map((a) => a[i])]);
}

function softmax(logits) {
  let max = Math.max(...logits);
  let scores = logits.map((x) => Math.exp(x - max));
  let sum = scores.reduce((a, b) => a + b, 0);
  return scores.map((s) => s / sum);
}
