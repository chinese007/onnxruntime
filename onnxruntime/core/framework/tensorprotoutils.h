// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <type_traits>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/framework/ml_value.h"
#include "core/framework/path_lib.h"
#include "core/graph/onnx_protobuf.h"

namespace ONNX_NAMESPACE {
class TensorProto;
class TensorShapeProto;
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
class Tensor;
namespace utils {
std::vector<int64_t> GetTensorShapeFromTensorShapeProto(const ONNX_NAMESPACE::TensorShapeProto& tensor_shape_proto);
common::Status TensorProtoToMLValue(const std::basic_string<PATH_CHAR_TYPE>& tensor_proto_path,
                                    const ONNX_NAMESPACE::TensorProto& input, AllocatorPtr allocator,
                                    void* preallocated, size_t preallocated_size, MLValue& value);
// This function doesn't support string tensors
ONNX_NAMESPACE::TensorProto::DataType GetTensorProtoType(const Tensor& tensor);

//How much memory it will need for putting the content of this tensor into a plain array
//string/complex64/complex128 tensors are not supported.
//The output value could be zero or -1.
template <size_t alignment>
common::Status GetSizeInBytesFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out);

//This is a private function. It is exposed here only for testing.
template <typename T>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, /*out*/ T* p_data, int64_t expected_size);
}  // namespace utils
}  // namespace onnxruntime
