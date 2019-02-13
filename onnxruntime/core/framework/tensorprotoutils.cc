// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"

#include <memory>
#include <algorithm>
#include <limits>
#include <gsl/pointers>

#include "path_lib.h"
#include "core/common/logging/logging.h"
#include "core/graph/onnx_protobuf.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/framework/ml_value_patterns_planner.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;

namespace {
class ExternalDataInfo {
 private:
  std::string rel_path;
  //-1 means doesn't exist
  ptrdiff_t offset;
  //-1 means doesn't exist
  ptrdiff_t length;
  std::string checksum;

 public:
  std::string GetRelPath() const {
    return rel_path;
  }

  ptrdiff_t GetOffset() const {
    return offset;
  }
  ptrdiff_t GetLength() const {
    return length;
  }

  const std::string& GetChecksum() const {
    return checksum;
  }

  //If the value of 'offset' or 'length' field is larger the max value of ssize_t, this function will treat it as a wrong value and return FAIL.
  static Status Create(const ::google::protobuf::RepeatedPtrField<::onnx::StringStringEntryProto>& input, std::unique_ptr<ExternalDataInfo>* out) {
    *out = std::make_unique<ExternalDataInfo>();
    for (int i = 0; i != input.size(); ++i) {
      StringStringEntryProto stringmap = input[i];
      if (!stringmap.has_key() || !stringmap.has_value())
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error!");
      if (stringmap.key() == "location" && !stringmap.value().empty()) {
        (*out)->rel_path = stringmap.value();
      } else if (stringmap.key() == "offset" && !stringmap.value().empty()) {
        char* end;
        (*out)->offset = MyStrtoPtrDiff(stringmap.value().c_str(), &end, 10);
        if (end != stringmap.value().c_str() + stringmap.value().length())
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "parsing ", stringmap.value(), " failed");
      } else if (stringmap.key() == "length" && !stringmap.value().empty()) {
        char* end;
        (*out)->length = MyStrtoPtrDiff(stringmap.value().c_str(), &end, 10);
        if (end != stringmap.value().c_str() + stringmap.value().length())
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "parsing ", stringmap.value(), " failed");
      } else if (stringmap.key() == "checksum" && !stringmap.value().empty()) {
        (*out)->checksum = stringmap.value();
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error!");
      }
    }
    if ((*out)->rel_path.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error! Missing 'location'");
    }
    return Status::OK();
  }
};
#ifdef __GNUC__
constexpr inline bool IsLittleEndianOrder() noexcept {
  return __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
}
#else
GSL_SUPPRESS(type .1)  // allow use of reinterpret_cast for this special case
constexpr inline bool IsLittleEndianOrder() noexcept {
  static int n = 1;
  return (*reinterpret_cast<char*>(&n) == 1);
}
#endif

std::vector<int64_t> GetTensorShapeFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  const auto& dims = tensor_proto.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = dims[i];
  }

  return tensor_shape_vec;
}
}  // namespace

namespace onnxruntime {
namespace utils {

template <typename T>
static void UnpackTensorWithRawData(const void* raw_data, size_t raw_data_length, /*out*/ T* p_data) {
  // allow this low level routine to be somewhat unsafe. assuming it's thoroughly tested and valid
  GSL_SUPPRESS(type)       // type.1 reinterpret-cast; type.4 C-style casts; type.5 'T result;' is uninitialized;
  GSL_SUPPRESS(bounds .1)  // pointer arithmetic
  GSL_SUPPRESS(f .23)      // buff and temp_bytes never tested for nullness and could be gsl::not_null
  {
    const size_t type_size = sizeof(T);

    if (IsLittleEndianOrder()) {
      memcpy(p_data, raw_data, raw_data_length);
    } else {
      const char* buff = reinterpret_cast<const char*>(raw_data);
      for (size_t i = 0; i < raw_data_length; i += type_size, buff += type_size) {
        T result;
        const char* temp_bytes = reinterpret_cast<char*>(&result);
        for (size_t j = 0; j < type_size; ++j) {
          memcpy((void*)&temp_bytes[j], (void*)&buff[type_size - 1 - i], 1);
        }
        p_data[i] = result;
      }
    }
  }
}

//This macro doesn't work for Float16/bool/string tensors
#define DEFINE_UNPACK_TENSOR(T, Type, field_name, field_size)                                                      \
  template <>                                                                                                      \
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, /*out*/ T* p_data, int64_t expected_size) {       \
    if (nullptr == p_data) {                                                                                       \
      const size_t size = tensor.has_raw_data() ? tensor.raw_data().size() : tensor.field_size();                  \
      if (size == 0)                                                                                               \
        return Status::OK();                                                                                       \
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);                                                \
    }                                                                                                              \
    if (nullptr == p_data || Type != tensor.data_type()) {                                                         \
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);                                                \
    }                                                                                                              \
    if (tensor.has_raw_data()) {                                                                                   \
      size_t expected_size_in_bytes;                                                                               \
      if (!IAllocator::CalcMemSizeForArray(expected_size, sizeof(T), &expected_size_in_bytes)) {                   \
        return Status(common::ONNXRUNTIME, common::FAIL, "size overflow");                                         \
      }                                                                                                            \
      if (tensor.raw_data().size() != expected_size_in_bytes)                                                      \
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,                                                                  \
                               "UnpackTensor: the pre-allocated size does not match the raw data size, expected ", \
                               expected_size_in_bytes, ", got ", tensor.raw_data().size());                        \
      UnpackTensorWithRawData(tensor.raw_data().data(), tensor.raw_data().size(), p_data);                         \
      return Status::OK();                                                                                         \
    }                                                                                                              \
    if (tensor.field_size() != expected_size)                                                                      \
      return Status(common::ONNXRUNTIME, common::FAIL,                                                             \
                    "UnpackTensor: the pre-allocated size does not match the size in proto");                      \
    auto& data = tensor.field_name();                                                                              \
    for (auto data_iter = data.cbegin(); data_iter != data.cend(); ++data_iter)                                    \
      *p_data++ = *reinterpret_cast<const T*>(data_iter);                                                          \
    return Status::OK();                                                                                           \
  }

//TODO: complex64 complex128
DEFINE_UNPACK_TENSOR(float, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, float_data, float_data_size)
DEFINE_UNPACK_TENSOR(double, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, double_data, double_data_size);
DEFINE_UNPACK_TENSOR(uint8_t, ONNX_NAMESPACE::TensorProto_DataType_UINT8, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int8_t, ONNX_NAMESPACE::TensorProto_DataType_INT8, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int16_t, ONNX_NAMESPACE::TensorProto_DataType_INT16, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(uint16_t, ONNX_NAMESPACE::TensorProto_DataType_UINT16, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int32_t, ONNX_NAMESPACE::TensorProto_DataType_INT32, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int64_t, ONNX_NAMESPACE::TensorProto_DataType_INT64, int64_data, int64_data_size)
DEFINE_UNPACK_TENSOR(uint64_t, ONNX_NAMESPACE::TensorProto_DataType_UINT64, uint64_data, uint64_data_size)
DEFINE_UNPACK_TENSOR(uint32_t, ONNX_NAMESPACE::TensorProto_DataType_UINT32, uint64_data, uint64_data_size)

template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor,
                    /*out*/ std::string* p_data,
                    int64_t expected_size) {
  if (nullptr == p_data) {
    if (tensor.string_data_size() == 0)
      return Status::OK();
    else
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_STRING != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (tensor.string_data_size() != expected_size)
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  auto& string_data = tensor.string_data();
  for (auto iter = string_data.cbegin(); iter != string_data.cend(); ++iter) {
    *p_data++ = *iter;
  }

  return Status::OK();
}
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor,
                    /*out*/ bool* p_data,
                    int64_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = tensor.has_raw_data() ? tensor.raw_data().size() : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_BOOL != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (tensor.has_raw_data()) {
    if (tensor.raw_data().size() != (expected_size) * sizeof(bool))
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "UnpackTensor: the pre-allocate size does not match the raw data size");

    UnpackTensorWithRawData(tensor.raw_data().data(), tensor.raw_data().size(), p_data);
    return Status::OK();
  }

  if (tensor.int32_data_size() != expected_size)
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");
  for (auto iter = tensor.int32_data().cbegin(); iter != tensor.int32_data().cend(); ++iter) {
    *p_data++ = static_cast<bool>(*iter);
  }

  return Status::OK();
}
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor,
                    /*out*/ MLFloat16* p_data,
                    int64_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = tensor.has_raw_data() ? tensor.raw_data().size() : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (tensor.has_raw_data()) {
    if (tensor.raw_data().size() != (expected_size) * sizeof(uint16_t))
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "UnpackTensor: the pre-allocate size does not match the raw data size");

    UnpackTensorWithRawData(tensor.raw_data().data(), tensor.raw_data().size(), p_data);
    return Status::OK();
  }

  if (tensor.int32_data_size() != expected_size)
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  const int max_value = std::numeric_limits<uint16_t>::max();
  for (int i = 0; i < static_cast<int>(expected_size); i++) {
    int v = tensor.int32_data()[i];
    if (v < 0 || v > max_value) {
      return Status(common::ONNXRUNTIME, common::FAIL, "data overflow");
    }
    p_data[i] = MLFloat16(static_cast<uint16_t>(v));
  }

  return Status::OK();
}

template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor,
                    /*out*/ BFloat16* p_data,
                    int64_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = tensor.has_raw_data() ? tensor.raw_data().size() : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();
    else
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16 != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (tensor.has_raw_data()) {
    if (tensor.raw_data().size() != (expected_size) * sizeof(uint16_t))
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "UnpackTensor: the pre-allocate size does not match the raw data size");

    UnpackTensorWithRawData(tensor.raw_data().data(), tensor.raw_data().size(), p_data);
    return Status::OK();
  }

  if (tensor.int32_data_size() != expected_size)
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  const int max_value = std::numeric_limits<uint16_t>::max();
  for (int i = 0; i < static_cast<int>(expected_size); i++) {
    int v = tensor.int32_data()[i];
    if (v < 0 || v > max_value) {
      return Status(common::ONNXRUNTIME, common::FAIL, "data overflow");
    }
    p_data[i] = BFloat16(static_cast<uint16_t>(v));
  }

  return Status::OK();
}

#define CASE_PROTO_TRACE(X, Y)                                                            \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X:                    \
    if (!IAllocator::CalcMemSizeForArrayWithAlignment<alignment>(size, sizeof(Y), out)) { \
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Invalid TensorProto");    \
    }                                                                                     \
    break;

template <size_t alignment>
common::Status GetSizeInBytesFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out) {
  const auto& dims = tensor_proto.dims();
  size_t size = 1;
  for (int i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0) {
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Invalid TensorProto");
    }
    if (!IAllocator::CalcMemSizeForArray(size, static_cast<size_t>(dims[i]), &size)) {
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Invalid TensorProto");
    }
  }
  switch (tensor_proto.data_type()) {
    CASE_PROTO_TRACE(FLOAT, float);
    CASE_PROTO_TRACE(DOUBLE, double);
    CASE_PROTO_TRACE(BOOL, bool);
    CASE_PROTO_TRACE(INT8, int8_t);
    CASE_PROTO_TRACE(INT16, int16_t);
    CASE_PROTO_TRACE(INT32, int32_t);
    CASE_PROTO_TRACE(INT64, int64_t);
    CASE_PROTO_TRACE(UINT8, uint8_t);
    CASE_PROTO_TRACE(UINT16, uint16_t);
    CASE_PROTO_TRACE(UINT32, uint32_t);
    CASE_PROTO_TRACE(UINT64, uint64_t);
    CASE_PROTO_TRACE(FLOAT16, MLFloat16);
    CASE_PROTO_TRACE(BFLOAT16, BFloat16);
    CASE_PROTO_TRACE(STRING, std::string);
    default:
      return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
  }
  return Status::OK();
}

std::vector<int64_t> GetTensorShapeFromTensorShapeProto(const ONNX_NAMESPACE::TensorShapeProto& tensor_shape_proto) {
  const auto& dims = tensor_shape_proto.dim();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = dims[i].has_dim_param()
                              ? -1 /* symbolic dimensions are represented as -1 in onnxruntime*/
                              : dims[i].dim_value();
  }
  return tensor_shape_vec;
}

template <typename T>
common::Status GetTensorByTypeFromTensorProto(const TensorProto& tensor_proto,
                                              const TensorShape& tensor_shape,
                                              std::unique_ptr<Tensor>* p_tensor,
                                              AllocatorPtr alloc,
                                              void* preallocated,
                                              size_t preallocated_size) {
  int64_t tensor_size = tensor_shape.Size();
  //tensor_size could be zero. see test_slice_start_out_of_bounds\test_data_set_0\output_0.pb
  if (tensor_size < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid shape ", tensor_shape);
  }
  size_t size_to_allocate;
  if (!IAllocator::CalcMemSizeForArrayWithAlignment<256>(static_cast<size_t>(tensor_size), sizeof(T),
                                                         &size_to_allocate)) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "size overflow");
  }

  if (preallocated && preallocated_size != size_to_allocate)
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The buffer planner is not consistent with tensor buffer size, expected ",
                           size_to_allocate, ", got ", preallocated_size);

  std::unique_ptr<Tensor> t;
  if (preallocated) {
    t = std::make_unique<Tensor>(DataTypeImpl::GetType<T>(), tensor_shape, preallocated, alloc->Info());
  } else {
    t = std::make_unique<Tensor>(DataTypeImpl::GetType<T>(), tensor_shape, alloc);
  }
  ORT_RETURN_IF_ERROR(::onnxruntime::utils::UnpackTensor(tensor_proto, t->MutableData<T>(), tensor_size));
  *p_tensor = std::move(t);
  return common::Status::OK();
}



#define CASE_PROTO(X, Y)                                                                                                                       \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X:                                                                         \
    ORT_RETURN_IF_ERROR(GetTensorByTypeFromTensorProto<Y>(tensor_proto, tensor_shape, &p_tensor, allocator, preallocated, preallocated_size)); \
    break;

Status TensorProtoToMLValue(const std::basic_string<PATH_CHAR_TYPE>& tensor_proto_path,
                            const ONNX_NAMESPACE::TensorProto& tensor_proto, AllocatorPtr allocator, void* preallocated,
                            size_t preallocated_size, MLValue& value) {
  std::unique_ptr<Tensor> p_tensor;
  {
    std::vector<int64_t> tensor_shape_vec = GetTensorShapeFromTensorProto(tensor_proto);
    // Note: We permit an empty tensor_shape_vec, and treat it as a scalar (a tensor of size 1).
    TensorShape tensor_shape{tensor_shape_vec};
    if (tensor_proto.data_location() == TensorProto_DataLocation_EXTERNAL) {
      std::unique_ptr<ExternalDataInfo> external_data_info;
      ORT_RETURN_IF_ERROR(ExternalDataInfo::Create(tensor_proto.external_data(), &external_data_info));
      if (external_data_info->GetOffset() > 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Cannot support tensor data with offset > 0");
      }
      // TODO:load the file
      (void)tensor_proto_path;
    }
    switch (tensor_proto.data_type()) {
      CASE_PROTO(FLOAT, float);
      CASE_PROTO(DOUBLE, double);
      CASE_PROTO(BOOL, bool);
      CASE_PROTO(INT8, int8_t);
      CASE_PROTO(INT16, int16_t);
      CASE_PROTO(INT32, int32_t);
      CASE_PROTO(INT64, int64_t);
      CASE_PROTO(UINT8, uint8_t);
      CASE_PROTO(UINT16, uint16_t);
      CASE_PROTO(UINT32, uint32_t);
      CASE_PROTO(UINT64, uint64_t);
      CASE_PROTO(STRING, std::string);
      CASE_PROTO(FLOAT16, MLFloat16);
      CASE_PROTO(BFLOAT16, BFloat16);
      default: {
        std::ostringstream ostr;
        ostr << "Initialized tensor with unexpected type: " << tensor_proto.data_type();
        return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
      }
    }
  }
  value.Init(p_tensor.release(),
             DataTypeImpl::GetType<Tensor>(),
             DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  return Status::OK();
}

TensorProto::DataType GetTensorProtoType(const Tensor& tensor) {
  auto tensor_type = tensor.DataType();
  TensorProto::DataType dtype = TensorProto_DataType_UNDEFINED;

  if (tensor_type == DataTypeImpl::GetType<float>())
    dtype = TensorProto_DataType_FLOAT;
  else if (tensor_type == DataTypeImpl::GetType<double>())
    dtype = TensorProto_DataType_DOUBLE;
  else if (tensor_type == DataTypeImpl::GetType<int8_t>())
    dtype = TensorProto_DataType_INT8;
  else if (tensor_type == DataTypeImpl::GetType<int16_t>())
    dtype = TensorProto_DataType_INT16;
  else if (tensor_type == DataTypeImpl::GetType<int32_t>())
    dtype = TensorProto_DataType_INT32;
  else if (tensor_type == DataTypeImpl::GetType<int64_t>())
    dtype = TensorProto_DataType_INT64;
  else if (tensor_type == DataTypeImpl::GetType<uint8_t>())
    dtype = TensorProto_DataType_UINT8;
  else if (tensor_type == DataTypeImpl::GetType<uint16_t>())
    dtype = TensorProto_DataType_UINT16;
  else if (tensor_type == DataTypeImpl::GetType<uint32_t>())
    dtype = TensorProto_DataType_UINT32;
  else if (tensor_type == DataTypeImpl::GetType<uint64_t>())
    dtype = TensorProto_DataType_UINT64;
  else if (tensor_type == DataTypeImpl::GetType<bool>())
    dtype = TensorProto_DataType_BOOL;
  else if (tensor_type == DataTypeImpl::GetType<MLFloat16>())
    dtype = TensorProto_DataType_FLOAT16;
  else if (tensor_type == DataTypeImpl::GetType<BFloat16>())
    dtype = TensorProto_DataType_BFLOAT16;

  return dtype;
}

template common::Status GetSizeInBytesFromTensorProto<256>(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out);
}  // namespace utils
}  // namespace onnxruntime
