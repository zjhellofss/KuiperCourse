//
// Created by fss on 23-1-1.
//
#include <glog/logging.h>
#include "layer/maxpooling_layer.hpp"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
MaxPoolingLayer::MaxPoolingLayer(const std::shared_ptr<Operator> &op) : Layer("maxpooling") {
  CHECK(op->op_type_ == OpType::kOperatorMaxPooling) << "Operator has a wrong type: " << int(op->op_type_);
  MaxPoolingOp *max_pooling_op = dynamic_cast<MaxPoolingOp *>(op.get());

  CHECK(max_pooling_op != nullptr) << "MaxPooling operator is empty";
  this->op_ = std::make_unique<MaxPoolingOp>(*max_pooling_op);
}

void MaxPoolingLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                               std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  CHECK(this->op_ != nullptr); // 判断operator是否为空，是否类型正确
  CHECK(this->op_->op_type_ == OpType::kOperatorMaxPooling);
  CHECK(!inputs.empty());
  // 获取到池化相关的属性
  const uint32_t padding_h = this->op_->padding_height();
  const uint32_t padding_w = this->op_->padding_width();
  const uint32_t kernel_h = this->op_->pooling_height();
  const uint32_t kernel_w = this->op_->pooling_width();
  const uint32_t stride_h = this->op_->stride_height();
  const uint32_t stride_w = this->op_->stride_width();

  // 取得输入的批次数量
  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input_data_ = inputs.at(i)->Clone();
    // 如果padding_h padding_w不为0的话，就做一个填充操作，周围填充一圈负无穷
    input_data_->Padding({padding_h, padding_h, padding_w, padding_w}, std::numeric_limits<float>::lowest());
    // 获得输入特征图的大小、通道数量
    const uint32_t input_h = input_data_->rows();
    const uint32_t input_w = input_data_->cols();
    const uint32_t input_c = input_data_->channels();
    const uint32_t output_c = input_c;

    // input_h 输入的高度
    // input_w 输入的宽度
    // kernel_h 窗口的高度
    // kernel_w 窗口的宽度
    // 计算输出特征图的大小
    const uint32_t output_h = uint32_t(std::floor((input_h - kernel_h) / stride_h + 1));
    const uint32_t output_w = uint32_t(std::floor((input_w - kernel_w) / stride_w + 1));
    CHECK(output_w > 0 && output_h > 0);

    std::shared_ptr<Tensor<float>> output_data = std::make_shared<Tensor<float>>(output_c, output_h, output_w);
    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat &input_channel = input_data_->at(ic);
      // 池化操作也是逐个通道进行的
      arma::fmat &output_channel = output_data->at(ic);

      // 计算过程 行上的
      for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h) {
        // 列上的
        for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w) {
          // 已知窗口开始位置和结束位置的时候，取得这一块区域
          const arma::fmat &region = input_channel.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
          // 取合围范围内的最大值
          output_channel.at(int(r / stride_h), int(c / stride_w)) = region.max();
        }
      }
    }
    outputs.push_back(output_data);
  }
}

std::shared_ptr<Layer> MaxPoolingLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
  CHECK(op->op_type_ == OpType::kOperatorMaxPooling);
  std::shared_ptr<Layer> max_layer = std::make_shared<MaxPoolingLayer>(op);
  return max_layer;
}

LayerRegistererWrapper kMaxPoolingLayer(OpType::kOperatorMaxPooling, MaxPoolingLayer::CreateInstance);
}