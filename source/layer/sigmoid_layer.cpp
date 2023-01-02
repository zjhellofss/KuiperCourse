//
// Created by fss on 22-12-29.
//
#include <glog/logging.h>
#include "layer/sigmoid_layer.hpp"
#include "ops/relu_op.hpp"
#include "factory/layer_factory.hpp"
#include <armadillo>

namespace kuiper_infer {

SigmoidLayer::SigmoidLayer(const std::shared_ptr<Operator> &op) : Layer("Sigmoid") {
  SigmoidOperator *sigmoid_op = dynamic_cast<SigmoidOperator *>(op.get());

  CHECK(sigmoid_op != nullptr) << "Sigmoid operator is empty";
  this->op_ = std::make_unique<SigmoidOperator>();
}

void SigmoidLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                            std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  CHECK(this->op_ != nullptr);
  CHECK(this->op_->op_type_ == OpType::kOperatorSigmoid);
  CHECK(!inputs.empty());

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
    std::shared_ptr<Tensor<float>> output_data = input_data->Clone();
//补充,y=1/(1+e^{-x})
// 参考relu_layer来
// 或者有自己的想法也可以
    outputs.push_back(output_data);
  }
}
std::shared_ptr<Layer> SigmoidLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
  CHECK(op != nullptr);
  CHECK(op->op_type_ == OpType::kOperatorSigmoid);

  std::shared_ptr<Layer> sigmoid_layer = std::make_shared<SigmoidLayer>(op);
  return sigmoid_layer;
}
// OpType::kOperatorRelu 自己替换掉,换成sigmoid对应类型
//LayerRegistererWrapper kReluLayer1(OpType::kOperatorRelu, SigmoidLayer::CreateInstance);
}