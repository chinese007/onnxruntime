#include "cuda_common.h"
#include "cuda_execution_provider.h"
#include "core/framework/transformer_memcpy.h"
#include "core/framework/memcpy.h"
#include "cuda_fence.h"
#include "cuda_allocator.h"
#include "core/framework/kernel_registry.h"

using namespace LotusIR;
using namespace Lotus::Common;

namespace Lotus {

namespace Cuda {

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<kMemTypeCPUInput>(0)
        .ExecQueueId(kCudaStreamCopyIn)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<kMemTypeCPUOutput>(0)
        .ExecQueueId(kCudaStreamCopyOut)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

}  // namespace Cuda

thread_local std::shared_ptr<CUDAExecutionProvider::PerThreadContext> CUDAExecutionProvider::per_thread_context_;
thread_local AllocatorPtr CUDAExecutionProvider::per_thread_default_allocator_;

CUDAExecutionProvider::PerThreadContext::PerThreadContext(int device_id) {
  CUDA_CALL_THROW(cudaSetDevice(device_id));
  CUBLAS_CALL_THROW(cublasCreate(&cublas_handle_));
  CUDNN_CALL_THROW(cudnnCreate(&cudnn_handle_));
}

CUDAExecutionProvider::PerThreadContext::~PerThreadContext() {
  CUBLAS_CALL_THROW(cublasDestroy(cublas_handle_));
  CUDNN_CALL_THROW(cudnnDestroy(cudnn_handle_));
}

CUDAExecutionProvider::CUDAExecutionProvider(const CUDAExecutionProviderInfo& info)
    : device_id_(info.device_id) {
  CUDA_CALL_THROW(cudaSetDevice(device_id_));
  // create streams, default is nullptr
  streams_[kCudaStreamDefault] = nullptr;
  CUDA_CALL_THROW(cudaStreamCreateWithFlags(&streams_[kCudaStreamCopyIn], cudaStreamNonBlocking));
  CUDA_CALL_THROW(cudaStreamCreateWithFlags(&streams_[kCudaStreamCopyOut], cudaStreamNonBlocking));

  DeviceAllocatorRegistrationInfo default_allocator_info({kMemTypeDefault,
                                                          [](int id) { return std::make_unique<CUDAAllocator>(id); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(kMemTypeDefault, CreateAllocator(default_allocator_info, device_id_));

  DeviceAllocatorRegistrationInfo pinned_allocator_info({kMemTypeCPUOutput,
                                                         [](int) { return std::make_unique<CUDAPinnedAllocator>(); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(kMemTypeCPUOutput, CreateAllocator(pinned_allocator_info, device_id_));
}

CUDAExecutionProvider::~CUDAExecutionProvider() {
  auto cpu_alloc = GetAllocator(kMemTypeCPU);
  std::lock_guard<std::mutex> lock(deferred_release_cpu_ptr_mutex_);
  auto it = deferred_release_cpu_ptr_.begin();
  while (it != deferred_release_cpu_ptr_.end()) {
    auto& e = it->first;
    auto& v = it->second;
    if (v.recorded)
      CUDA_CALL_THROW(cudaEventSynchronize(e));
    for (auto p : v.cpu_ptrs) {
      cpu_alloc->Free(p);
    }
    CUDA_CALL_THROW(cudaEventDestroy(e));
    it = deferred_release_cpu_ptr_.erase(it);
  }
  CUDA_CALL_THROW(cudaStreamDestroy(streams_[kCudaStreamCopyIn]));
  CUDA_CALL_THROW(cudaStreamDestroy(streams_[kCudaStreamCopyOut]));

  ReleasePerThreadStuffs();
}

void CUDAExecutionProvider::ReleasePerThreadStuffs() const {
  if (per_thread_default_allocator_) {
    std::lock_guard<std::mutex> lock(default_allocator_pool_mutex_);
    default_allocator_pool_.push_back(per_thread_default_allocator_);
    per_thread_default_allocator_.reset();
  }
  if (per_thread_context_) {
    std::lock_guard<std::mutex> lock(context_pool_mutex_);
    context_pool_.push_back(per_thread_context_);
    per_thread_context_.reset();
  }
}

AllocatorPtr CUDAExecutionProvider::GetAllocator(MemType mem_type) const {
  // Pinned memory allocator is shared between threads, but CUDA memory allocator is per-thread or it may cause result changes
  // A hypothesis is that arena allocator is not aligned with CUDA output cache, and data from different kernel writes may
  // cause cacheline to contain dirty data.
  if (mem_type == kMemTypeDefault) {
    if (!per_thread_default_allocator_) {
      std::lock_guard<std::mutex> lock(default_allocator_pool_mutex_);
      if (default_allocator_pool_.empty()) {
        DeviceAllocatorRegistrationInfo default_allocator_info(
            {kMemTypeDefault,
             [](int id) { return std::make_unique<CUDAAllocator>(id); }, std::numeric_limits<size_t>::max()});
        per_thread_default_allocator_ = CreateAllocator(default_allocator_info, device_id_);
      } else {
        per_thread_default_allocator_ = default_allocator_pool_.back();
        default_allocator_pool_.pop_back();
      }
    }
    return per_thread_default_allocator_;
  } else {
    return IExecutionProvider::GetAllocator(mem_type);
  }
}

Status CUDAExecutionProvider::Sync() {
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
  return Status::OK();
}

void CUDAExecutionProvider::AddDeferredReleaseCPUPtr(void* p) {
  // when not running in InferenceSession (e.g. Test)
  // it's OK to not remember the deferred release ptr
  // as the actual memory will be cleaned in arena allocator dtor
  auto current_deferred_release_event = per_thread_context_->GetCurrentDeferredReleaseEvent();
  if (current_deferred_release_event) {
    std::lock_guard<std::mutex> lock(deferred_release_cpu_ptr_mutex_);
    auto iter = deferred_release_cpu_ptr_.find(current_deferred_release_event);
    LOTUS_ENFORCE(iter != deferred_release_cpu_ptr_.end());
    iter->second.cpu_ptrs.push_back(p);
  }
}

Status CUDAExecutionProvider::OnRunStart() {
  auto cpu_alloc = GetAllocator(kMemTypeCPU);
  // check if cudaEvents has passed for deferred release
  // note that we need to take a mutex in case of multi-threaded Run()
  std::lock_guard<std::mutex> lock(deferred_release_cpu_ptr_mutex_);
  auto it = deferred_release_cpu_ptr_.begin();
  while (it != deferred_release_cpu_ptr_.end()) {
    auto& e = it->first;
    auto& v = it->second;
    // note that cudaEventQuery returns cudaSucess before first cudaEventRecord
    if (v.recorded && cudaSuccess == cudaEventQuery(e)) {
      for (auto p : v.cpu_ptrs) {
        cpu_alloc->Free(p);
      }
      cudaEvent_t expired_event = it->first;
      it = deferred_release_cpu_ptr_.erase(it);
      CUDA_RETURN_IF_ERROR(cudaEventDestroy(expired_event));
    } else {
      ++it;
    }
  }
  // start a new per_thread context, store in TLS
  {
    std::lock_guard<std::mutex> ctx_lock(context_pool_mutex_);
    if (context_pool_.empty()) {
      per_thread_context_ = std::make_shared<PerThreadContext>(device_id_);
    } else {
      per_thread_context_ = context_pool_.back();
      context_pool_.pop_back();
    }
  }
  auto& current_deferred_release_event = per_thread_context_->GetCurrentDeferredReleaseEvent();
  CUDA_RETURN_IF_ERROR(cudaEventCreate(&current_deferred_release_event, cudaEventDisableTiming));
  deferred_release_cpu_ptr_.emplace(current_deferred_release_event, DeferredReleaseCPUPtrs());
  return Status::OK();
}

Status CUDAExecutionProvider::OnRunEnd() {
  LOTUS_RETURN_IF_NOT(per_thread_context_ != nullptr);
  // record deferred release event on default stream, and release per_thread_context
  auto current_deferred_release_event = per_thread_context_->GetCurrentDeferredReleaseEvent();
  CUDA_RETURN_IF_ERROR(cudaEventRecord(current_deferred_release_event, nullptr));
  ReleasePerThreadStuffs();
  std::lock_guard<std::mutex> lock(deferred_release_cpu_ptr_mutex_);
  deferred_release_cpu_ptr_[current_deferred_release_event].recorded = true;
  return Status::OK();
}

Status CUDAExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) const {
  return CopyTensor(src, dst, kCudaStreamDefault);
}

Status CUDAExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  if (src.Shape().Size() != dst.Shape().Size()) {
    return Status(LOTUS, FAIL, "Tensor size mismatch");
  }

  if (src.Location().name != CUDA && src.Location().name != CUDA_PINNED &&
      dst.Location().name != CUDA && dst.Location().name != CUDA_PINNED) {
    return Status(LOTUS, FAIL, "Unsupported tensor location: src_location is: " + src.Location().name + " and dst_location is: " + dst.Location().name);
  }

  size_t bytes = src.DataType()->Size() * src.Shape().Size();

  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  if (dst.Location().name == CUDA) {
    if (src.Location().name == CUDA_PINNED) {
      // copy from pinned memory to GPU, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, streams_[exec_queue_id]));
    } else if (src.Location().name == CUDA) {
      // copying between GPU, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice, streams_[kCudaStreamDefault]));
    } else {
      // copy from other CPU memory to GPU, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice));
    }
  } else if (src.Location().name == CUDA) {
    if (dst.Location().name == CUDA_PINNED) {
      // copying from GPU to pinned memory, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, streams_[exec_queue_id]));
    } else {
      // copying from GPU to CPU memory, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost));
    }
  } else {
    // copying between cpu memory
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

namespace Cuda {
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyToHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MatMul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 4, Concat);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Unsqueeze);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Flatten);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Squeeze);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Identity);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, Dropout);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Gather);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Affine);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Affine);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Affine);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Elu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Elu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Elu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ParametricSoftplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ParametricSoftplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ParametricSoftplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ScaledTanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ScaledTanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ScaledTanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Selu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Selu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Selu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softsign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softsign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softsign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ThresholdedRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ThresholdedRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ThresholdedRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, And);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Or);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Xor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int32_t, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int64_t, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint32_t, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint64_t, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int8_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int16_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int32_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int64_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint8_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint16_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint32_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint64_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int8_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int16_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int32_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int64_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int8_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int16_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int32_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int64_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint8_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint16_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint32_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint64_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, bool, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, float, Pad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, double, Pad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, MLFloat16, Pad);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 5, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 4, Reshape_1);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Shape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Transpose);

static void RegisterCudaKernels(std::function<void(KernelCreateInfo&&)> fn) {
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, Gemm)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MatMul)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 4, Concat)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Unsqueeze)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Flatten)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Squeeze)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Identity)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, Dropout)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Gather)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Tile)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Tile)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Tile)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Affine)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Affine)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Affine)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Elu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Elu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Elu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, HardSigmoid)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, HardSigmoid)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, HardSigmoid)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, LeakyRelu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, LeakyRelu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, LeakyRelu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ParametricSoftplus)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ParametricSoftplus)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ParametricSoftplus)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Relu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Relu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Relu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ScaledTanh)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ScaledTanh)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ScaledTanh)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Selu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Selu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Selu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Sigmoid)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Sigmoid)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Sigmoid)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softsign)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softsign)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softsign)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Tanh)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Tanh)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Tanh)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ThresholdedRelu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ThresholdedRelu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ThresholdedRelu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softplus)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softplus)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softplus)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softmax)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softmax)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softmax)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Pow)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Pow)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Pow)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, PRelu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, PRelu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, PRelu)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, And)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Or)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Xor)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int32_t, Sum)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int64_t, Sum)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint32_t, Sum)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint64_t, Sum)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Sum)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Sum)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Sum)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Add)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Add)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Add)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Add)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Add)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Add)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Add)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Sub)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Sub)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Sub)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Sub)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Sub)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Sub)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Sub)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Mul)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Mul)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Mul)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Mul)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Mul)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Mul)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Mul)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Div)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Div)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Div)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Div)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Div)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Div)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Div)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int8_t, Abs)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int16_t, Abs)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int32_t, Abs)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int64_t, Abs)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint8_t, Abs)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint16_t, Abs)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint32_t, Abs)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint64_t, Abs)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Abs)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Abs)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Abs)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int8_t, Neg)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int16_t, Neg)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int32_t, Neg)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int64_t, Neg)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Neg)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Neg)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Neg)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Floor)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Floor)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Floor)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Ceil)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Ceil)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Ceil)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Reciprocal)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Reciprocal)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Reciprocal)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Sqrt)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Sqrt)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Sqrt)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Log)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Log)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Log)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Exp)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Exp)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Exp)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, BatchNormalization)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, BatchNormalization)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, BatchNormalization)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Conv)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Conv)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Conv)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ConvTranspose)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ConvTranspose)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ConvTranspose)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, AveragePool)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, AveragePool)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, GlobalAveragePool)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, MaxPool)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, MaxPool)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, GlobalMaxPool)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ArgMax)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ArgMax)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ArgMax)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ArgMin)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ArgMin)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ArgMin)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceL1)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceL1)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceL1)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceL2)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceL2)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceL2)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceMax)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceMax)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceMax)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceMean)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceMean)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceMean)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceMin)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceMin)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceMin)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceProd)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceProd)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceProd)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceSum)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceSum)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceSum)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Cast)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Cast)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Cast)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int8_t, Cast)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int16_t, Cast)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int32_t, Cast)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int64_t, Cast)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint8_t, Cast)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint16_t, Cast)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint32_t, Cast)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint64_t, Cast)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, bool, Cast)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, float, Pad)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, double, Pad)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, MLFloat16, Pad)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 5, Reshape)>());
  fn(BuildKernel<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 4, Reshape_1)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Shape)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Tile)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Tile)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Tile)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Transpose)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Transpose)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Transpose)>());
}

}  // namespace Cuda

std::shared_ptr<KernelRegistry> CUDAExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>(Lotus::Cuda::RegisterCudaKernels);
  return kernel_registry;
}

}  // namespace Lotus
