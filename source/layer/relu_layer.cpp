//
// Created by fss on 22-12-20.
//
#include <glog/logging.h>
#include "ops/relu_op.hpp"
#include "layer/relu_layer.hpp"

namespace kuiper_infer {
ReluLayer::ReluLayer(const std::shared_ptr<Operator> &op) : Layer("Relu") {
  CHECK(op->kOpType == OpType::kOperatorRelu) << "Operator has a wrong type: " << int(op->kOpType);
  ReluOperator *relu_op = dynamic_cast<ReluOperator *>(op.get());

  CHECK(relu_op != nullptr) << "Relu operator is empty";
  this->op_ = std::make_shared<ReluOperator>(relu_op->get_thresh());
}

void ReluLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                         std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  CHECK(this->op_ != nullptr);
  CHECK(this->op_->kOpType == OpType::kOperatorRelu);
  for (int i = 0; i < inputs.size(); ++i) {
    CHECK(!inputs.at(i)->empty());
    std::shared_ptr<Tensor<float>> input_data = inputs.at(i);
    input_data->data().transform([&](float value) {
      float thresh = op_->get_thresh();
      if (value >= thresh) {
        return value;
      } else {
        return 0.f;
      }
    });
    outputs.push_back(input_data);
  }
}
}