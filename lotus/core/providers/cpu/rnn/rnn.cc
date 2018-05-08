#include "core/providers/cpu/rnn/rnn.h"
#include "core/providers/cpu/rnn/rnn_activation_functors.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {
REGISTER_KERNEL(KernelDefBuilder("RNN")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("T1", DataTypeImpl::GetTensorType<int>()),
                RNN<float, int>);

template <typename T>
T Clip(const T& x, T clip) {
  if (clip < 0)
    return x;
  else
    return std::max(std::min(x, clip), -clip);
}

template <typename T>
void ApplyActivationToBatchs(const Tensor* sequence_lens, const T* h_prev, T* Y_buffer_data_current_frame,
                             int64_t time_step, int64_t batch_size, int64_t hidden_size,
                             T alpha, T beta, T clip, std::function<T(T, T, T)> activation_func) {
  for (int batch = 0; batch < batch_size; batch++) {
    bool valid = true;
    if (nullptr != sequence_lens) {
      valid = time_step < sequence_lens->Data<int>()[batch];
    }

    for (int feature = 0; feature < hidden_size; ++feature) {
      int64_t y_index = batch * hidden_size + feature;
      if (!valid) {
        // copy from previous time_step.
        // this will not happen at time_step 0.
        Y_buffer_data_current_frame[y_index] = h_prev[batch * hidden_size + feature];
      } else {
        Y_buffer_data_current_frame[y_index] = activation_func(
            Clip(Y_buffer_data_current_frame[y_index], clip), alpha, beta);
      }
    }
  }
}

template <typename T>
void Assign_Y_h(const T* Y_buffer_data, Tensor* Y_h, const Tensor* sequence_lens,
                int64_t num_directions, int direction, bool isReverse, int64_t batch_size, int64_t seq_length, int64_t hidden_size) {
  for (int batch = 0; batch < batch_size; batch++) {
    int64_t last_time_step = isReverse ? 0 : seq_length - 1;
    if (nullptr != sequence_lens && !isReverse)
      last_time_step = sequence_lens->Data<int>()[batch] - 1;
    int64_t y_offset = last_time_step * num_directions * batch_size * hidden_size +
                       direction * batch_size * hidden_size +
                       batch * hidden_size;
    int64_t Y_h_offset = direction * batch_size * hidden_size + batch * hidden_size;
    Math::CopyVector<T, CPUMathUtil>((const int)hidden_size, Y_buffer_data + y_offset,
                                     Y_h->MutableData<T>() + Y_h_offset,
                                     &CPUMathUtil::Instance());
  }
}

template <typename T>
void ClearMissingFrames(T* Y_buffer_data, const Tensor* sequence_lens,
                        int64_t num_directions, int64_t batch_size, int64_t seq_length, int64_t hidden_size) {
  for (int direction = 0; direction < num_directions; direction++) {
    for (int batch = 0; batch < batch_size; batch++) {
      if (sequence_lens->Data<int>()[batch] < seq_length) {
        for (int seq = sequence_lens->Data<int>()[batch]; seq < seq_length; seq++) {
          int64_t offset =
              seq * num_directions * batch_size * hidden_size +
              direction * batch_size * hidden_size +
              batch * hidden_size;
          Math::Set<T, CPUMathUtil>(hidden_size, 0, Y_buffer_data + offset, &CPUMathUtil::Instance());
        }
      }
    }
  }
}

template <typename T>
using EigenMatrixMapRowMajor = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <>
Status RNN<float, int>::Compute(OpKernelContext* ctx) const {
  // inputs
  const Tensor& X = *ctx->Input<Tensor>(0);
  const Tensor& W = *ctx->Input<Tensor>(1);
  const Tensor& R = *ctx->Input<Tensor>(2);

  // optional inputs
  const Tensor* B = ctx->Input<Tensor>(3);
  const Tensor* sequence_lens = ctx->Input<Tensor>(4);
  const Tensor* initial_h = ctx->Input<Tensor>(5);

  int64_t num_directions = direction_ == "bidirectional" ? 2 : 1;
  int64_t seq_length = X.Shape()[0];
  int64_t batch_size = X.Shape()[1];
  int64_t hidden_size = W.Shape()[1];
  int64_t input_size = X.Shape()[2];

  if (X.Shape().NumDimensions() != 3 || X.Shape()[0] != seq_length || X.Shape()[1] == batch_size && X.Shape()[2] != input_size)
    return Status(LOTUS, StatusCode::INVALID_ARGUMENT, "Input X has invalid dimentions");
  if (W.Shape().NumDimensions() != 3 || W.Shape()[0] != num_directions || W.Shape()[1] != hidden_size || W.Shape()[2] != input_size)
    return Status(LOTUS, StatusCode::INVALID_ARGUMENT, "Input W has invalid dimentions");
  if (R.Shape().NumDimensions() != 3 || R.Shape()[0] != num_directions || R.Shape()[1] != hidden_size || R.Shape()[2] != hidden_size)
    return Status(LOTUS, StatusCode::INVALID_ARGUMENT, "Input R has invalid dimentions");
  if (B != nullptr && (B->Shape().NumDimensions() != 2 || B->Shape()[0] != num_directions || B->Shape()[1] != 2 * hidden_size))
    return Status(LOTUS, StatusCode::INVALID_ARGUMENT, "Input B has invalid dimentions");
  if (sequence_lens != nullptr &&
      (1 != sequence_lens->Shape().NumDimensions() ||
       batch_size != sequence_lens->Shape()[0] ||
       std::any_of(
           sequence_lens->Data<int>(),
           sequence_lens->Data<int>() + sequence_lens->Shape().Size(),
           [seq_length](int len) { return len <= 0 || len > seq_length; })))
    return Status(LOTUS, StatusCode::INVALID_ARGUMENT, "Input sequence_lens has invalid dimentions");
  if (initial_h != nullptr &&
      (initial_h->Shape().NumDimensions() != 3 ||
       initial_h->Shape()[0] != num_directions ||
       initial_h->Shape()[1] != batch_size ||
       initial_h->Shape()[2] != hidden_size))
    return Status(LOTUS, StatusCode::INVALID_ARGUMENT, "Input initial_h has invalid dimentions");

  // RNN outputs are optional
  std::vector<int64_t> Y_dims({seq_length, num_directions, batch_size, hidden_size});
  Tensor* Y = nullptr;
  int outputIndex = 0;
  if (output_sequence_ != 0) {
    Y = ctx->Output(outputIndex, Y_dims);
    outputIndex++;
  }

  std::vector<int64_t> Y_h_dims({num_directions, batch_size, hidden_size});
  Tensor* Y_h = ctx->Output(outputIndex, Y_h_dims);

  auto& info = OpKernel::Allocator();
  auto& alloc = AllocatorManager::Instance().GetArena(info.name, info.id);

  // X * W^t, each direction has shape of [seq_length, batch_size, hidden_size]
  auto x_matmul_data = alloc.Alloc(sizeof(float) * seq_length * batch_size * hidden_size);
  BufferUniquePtr x_matmul_buffer(x_matmul_data, BufferDeleter(&alloc));
  float* x_matmul_w_buffer_data = static_cast<float*>(x_matmul_buffer.get());

  float* Y_buffer_data;
  void* Y_data;
  BufferUniquePtr Y_matmul_buffer;
  if (Y != nullptr)
    Y_buffer_data = Y->MutableData<float>();
  else {
    Y_data = alloc.Alloc(sizeof(float) * seq_length * num_directions * batch_size * hidden_size);
    Y_matmul_buffer = BufferUniquePtr(Y_data, BufferDeleter(&alloc));
    Y_buffer_data = static_cast<float*>(Y_matmul_buffer.get());
  }

  int64_t Y_frame_size = batch_size * hidden_size;

  for (int direction = 0; direction < num_directions; direction++) {
    auto activation_func = detail::GetFuncByName<float>(activations_[direction], "Tanh");
    bool isReverse = direction_ == "reverse" || direction == 1;

    if (B != nullptr) {
      EigenMatrixMapRowMajor<float>(x_matmul_w_buffer_data, seq_length * batch_size, hidden_size).rowwise() =
          ConstEigenVectorMap<float>(B->Data<float>() + direction * 2 * hidden_size, hidden_size).transpose() +
          ConstEigenVectorMap<float>(B->Data<float>() + direction * 2 * hidden_size + hidden_size, hidden_size).transpose();
    } else {
      Math::Set<float, CPUMathUtil>(seq_length * batch_size * hidden_size, 0, x_matmul_w_buffer_data, &CPUMathUtil::Instance());
    }

    // X * W[direction]^t + B
    Math::Gemm<float, CPUMathUtil>(
        CblasNoTrans,
        CblasTrans,
        static_cast<int>(seq_length * batch_size),
        static_cast<int>(hidden_size),
        static_cast<int>(input_size),
        1,
        X.template Data<float>(),
        W.template Data<float>() + direction * hidden_size * input_size,
        1,
        x_matmul_w_buffer_data,
        &CPUMathUtil::Instance());

    for (int64_t t = 0; t < seq_length; t++) {
      int64_t time_step = isReverse ? (seq_length - t - 1) : t;
      int64_t Y_frame_offset = (time_step * num_directions + direction) * Y_frame_size;
      float* Y_buffer_data_current_frame = Y_buffer_data + Y_frame_offset;
      auto y_frame_mat = EigenMatrixMapRowMajor<float>(Y_buffer_data_current_frame, batch_size, hidden_size);

      const float* h_prev = nullptr;
      if (t == 0) {
        if (initial_h != nullptr)
          h_prev = initial_h->Data<float>();
      } else {
        if (isReverse)
          h_prev = Y_buffer_data_current_frame + num_directions * Y_frame_size;
        else
          h_prev = Y_buffer_data_current_frame - num_directions * Y_frame_size;
      }

      if (h_prev != nullptr) {
        // H_t_1 * R[direction]^t
        Math::Gemm<float, CPUMathUtil>(
            CblasNoTrans,
            CblasTrans,
            static_cast<int>(batch_size),
            static_cast<int>(hidden_size),
            static_cast<int>(hidden_size),
            1,
            h_prev,
            R.Data<float>() + direction * hidden_size * hidden_size,
            0,
            Y_buffer_data_current_frame,
            &CPUMathUtil::Instance());
      } else {
        Math::Set<float, CPUMathUtil>(batch_size * hidden_size, 0, Y_buffer_data_current_frame, &CPUMathUtil::Instance());
      }

      // X[time_step] * W^t + H_t_1 * R^t
      y_frame_mat += EigenMatrixMapRowMajor<float>(&x_matmul_w_buffer_data[time_step * Y_frame_size], batch_size, hidden_size);

      // apply activation
      ApplyActivationToBatchs<float>(sequence_lens, h_prev, Y_buffer_data_current_frame,
                                     time_step, batch_size, hidden_size,
                                     activation_alpha_[direction], activation_beta_[direction], clip_, activation_func);
    }  // close sequence loop

    Assign_Y_h<float>(Y_buffer_data, Y_h, sequence_lens,
                      num_directions, direction, isReverse, batch_size, seq_length, hidden_size);
  }

  // Now the full sequence is completed. Set missing frames to zero.
  if (nullptr != sequence_lens) {
    ClearMissingFrames(Y_buffer_data, sequence_lens,
                       num_directions, batch_size, seq_length, hidden_size);
  }

  return Status::OK();
}
}  // namespace Lotus