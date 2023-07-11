//
// Created by fss on 22-12-29.
//
#include <glog/logging.h>
#include "layer/sigmoid_layer.hpp"
#include "ops/relu_op.hpp"
#include "factory/layer_factory.hpp"
#include <armadillo>

namespace kuiper_infer
{

  SigmoidLayer::SigmoidLayer(const std::shared_ptr<Operator> &op) : Layer("Sigmoid")
  {
    SigmoidOperator *sigmoid_op = dynamic_cast<SigmoidOperator *>(op.get());

    CHECK(sigmoid_op != nullptr) << "Sigmoid operator is empty";
    this->op_ = std::make_unique<SigmoidOperator>();
  }

  void SigmoidLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                              std::vector<std::shared_ptr<Tensor<float>>> &outputs)
  {
    CHECK(this->op_ != nullptr);
    CHECK(this->op_->op_type_ == OpType::kOperatorSigmoid);
    CHECK(!inputs.empty());

    const uint32_t batch_size = inputs.size();
    for (uint32_t i = 0; i < batch_size; ++i)
    {
      const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
      std::shared_ptr<Tensor<float>> output_data = input_data->Clone();
      // 补充
      //  对张量中的每一个元素进行运算，进行relu运算
      output_data->data().transform([&](float value)
                                    { return 1 / (1 + std::exp(-value)); });

      // 把结果y放在outputs中
      outputs.push_back(output_data);
    }
  }
  std::shared_ptr<Layer> SigmoidLayer::CreateInstance(const std::shared_ptr<Operator> &op)
  {
    CHECK(op != nullptr);
    CHECK(op->op_type_ == OpType::kOperatorSigmoid);

    std::shared_ptr<Layer> sigmoid_layer = std::make_shared<SigmoidLayer>(op);
    return sigmoid_layer;
  }

  LayerRegistererWrapper kSigmoidLayer(OpType::kOperatorSigmoid, SigmoidLayer::CreateInstance);

}