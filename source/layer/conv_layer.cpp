//
// Created by fss on 23-2-2.
//

#include "layer/conv_layer.hpp"
#include "ops/conv_op.hpp"
#include <glog/logging.h>

namespace kuiper_infer {
ConvolutionLayer::ConvolutionLayer(const std::shared_ptr<Operator> &op) : Layer("convolution") {
  CHECK(op != nullptr && op->op_type_ == OpType::kOperatorConvolution);
  ConvolutionOp *conv_op = dynamic_cast< ConvolutionOp *>(op.get());

  CHECK(conv_op != nullptr) << "Expression operator is empty";
  this->op_ = std::make_unique<ConvolutionOp>(*conv_op);
}

void ConvolutionLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                std::vector<std::shared_ptr<Tensor<float>>> &outputs) {

  CHECK(this->op_ != nullptr && this->op_->op_type_ == OpType::kOperatorConvolution);

  CHECK(!inputs.empty()) << "Input is empty!";
  CHECK(inputs.size() == outputs.size());
  // weights就是一个vector数组，他代表了卷积核，存放了多个卷积核
  const auto &weights = this->op_->weight();
  CHECK(!weights.empty());

  std::vector<std::shared_ptr<ftensor >> bias_;
  if (this->op_->is_use_bias()) {
    bias_ = this->op_->bias();
  }

  const uint32_t stride_h = this->op_->stride_h();
  const uint32_t stride_w = this->op_->stride_w();
  CHECK(stride_w > 0 && stride_h > 0);
  const uint32_t padding_h = this->op_->padding_h();
  const uint32_t padding_w = this->op_->padding_w();
  const uint32_t groups = this->op_->groups();

  // 输入特征图的数量
  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
            << "The input feature map of conv layer is empty";

    std::shared_ptr<Tensor<float>> input_;
    if (padding_h > 0 || padding_w > 0) {
      // 进行padding填充，对四周进行填充
      input_ = input->Clone();
      input_->Padding({padding_h, padding_h, padding_w, padding_w}, 0);
    } else {
      input_ = input;
    }

    const uint32_t input_w = input_->cols();
    const uint32_t input_h = input_->rows();
    const uint32_t input_c = input_->channels();
    const uint32_t kernel_count = weights.size(); // 卷积核的个数
    CHECK(kernel_count > 0) << "kernel count must greater than zero";

    uint32_t kernel_h = weights.at(0)->rows();
    uint32_t kernel_w = weights.at(0)->cols();
    CHECK(kernel_h > 0 && kernel_w > 0)
            << "The size of kernel size is less than zero";

    uint32_t output_h = uint32_t(std::floor((input_h - kernel_h) / stride_h + 1));
    uint32_t output_w = uint32_t(std::floor((input_w - kernel_w) / stride_w + 1));
    CHECK(output_h > 0 && output_w > 0)
            << "The size of the output feature map is less than zero";

    if (groups != 1) {
      CHECK(kernel_count % groups == 0);
      CHECK(input_c % groups == 0);
    }

    for (uint32_t k = 0; k < kernel_count; ++k) {
      const std::shared_ptr<Tensor<float>> &kernel = weights.at(k);
      CHECK(kernel->rows() == kernel_h);
      CHECK(kernel->cols() == kernel_w);
      CHECK(kernel->channels() == input_c / groups);
    }

    uint32_t row_len = kernel_w * kernel_h; // 9
    uint32_t col_len = output_h * output_w; // 4
    if (!col_len) {
      col_len = 1;
    }

    uint32_t input_c_group = input_c / groups;
    uint32_t kernel_count_group = kernel_count / groups; // 1

    for (uint32_t g = 0; g < groups; ++g) {

      std::vector<arma::fmat> kernel_matrix_arr(kernel_count_group);
      // vector是存放展开之后的卷积核，有几个卷积核kernel_matrix_group size就等于多少
      arma::fmat kernel_matrix_c(1, row_len * input_c_group); // 展开之后的卷积核

      // 假设有三个卷积核
      for (uint32_t k = 0; k < kernel_count_group; ++k) {
        // 得到一个卷积核，得到第k个卷积核
        const std::shared_ptr<Tensor<float>> &kernel =
            weights.at(k);
        for (uint32_t ic = 0; ic < input_c_group; ++ic) {
          // 处理一个卷积核,卷积核当中的通道数
          // kernel->at(ic) 第k个卷积核的第ic个通道
          memcpy(kernel_matrix_c.memptr() + row_len * ic,
                 kernel->at(ic).memptr(), row_len * sizeof(float));
        }
        LOG(INFO) << "kernel展开后: " << "\n" << kernel_matrix_c;
        kernel_matrix_arr.at(k) = kernel_matrix_c;
      }
      // 3*9,4
      arma::fmat input_matrix(input_c_group * row_len, col_len);
      for (uint32_t ic = 0; ic < input_c_group; ++ic) {
        const arma::fmat &input_channel = input_->at(ic + g * input_c_group);
        int current_col = 0;
        for (uint32_t w = 0; w < input_w - kernel_w + 1; w += stride_w) {
          for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h) {

            float *input_matrix_c_ptr =
                input_matrix.colptr(current_col) + ic * row_len;
            current_col += 1;

            for (uint32_t kw = 0; kw < kernel_w; ++kw) {
              // kw =0,1,2
              // w+kw来找到窗口的列，r来找到窗口的行，region_ptr是窗口开始的位置
              const float *region_ptr = input_channel.colptr(w + kw) + r;
              memcpy(input_matrix_c_ptr, region_ptr, kernel_h * sizeof(float));
              input_matrix_c_ptr += kernel_h;
            }
          }
        }
      }
      LOG(INFO) << "input展开后: " << "\n" << input_matrix;

      std::shared_ptr<Tensor<float>> output_tensor = outputs.at(i);
      if (output_tensor == nullptr || output_tensor->empty()) {
        output_tensor =
            std::make_shared<Tensor<float>>(kernel_count, output_h, output_w);
        outputs.at(i) = output_tensor;
      }

      CHECK(output_tensor->rows() == output_h && output_tensor->cols() == output_w &&
          output_tensor->channels() == kernel_count) << "The output size of convolution is error";

      std::vector<arma::fmat> outputs_matrix(kernel_count_group);
      // 取出一个卷积核进行矩阵乘法
      for (uint32_t k = 0; k < kernel_count_group; ++k) {
        LOG(INFO) << "\n" << kernel_matrix_arr.at(k); // 拿出第一个展开后的卷积核
        LOG(INFO) << "\n" << input_matrix; // input_matrix是展开后的输入特征图

        const arma::fmat &output = kernel_matrix_arr.at(k) * input_matrix;
        LOG(INFO) << "output: \n" << output;
        outputs_matrix.at(k) = output;
      }

      bool use_bias = this->op_->is_use_bias();
      for (uint32_t k = 0; k < kernel_count_group; ++k) {
        std::shared_ptr<Tensor<float>> bias;
        if (!bias_.empty() && use_bias) {
          bias = bias_.at(k);
        }
        // 1 * 4
        arma::fmat output = outputs_matrix.at(k);
        LOG(INFO) << "输出的col" << output.n_cols;
        LOG(INFO) << "输出的row" << output.n_rows;

        CHECK(output.size() == output_h * output_w);

        output.reshape(output_h, output_w);
        if (bias != nullptr) {
          float bias_value = bias->index(0);
          output += bias_value;
        }
        output_tensor->at(k + g * kernel_count_group) = std::move(output);
      }
    }
  }
}
}