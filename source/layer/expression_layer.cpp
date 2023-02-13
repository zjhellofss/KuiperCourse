//
// Created by fss on 23-1-22.
//

#include <stack>
#include <glog/logging.h>
#include "layer/expression_layer.hpp"
namespace kuiper_infer {
ExpressionLayer::ExpressionLayer(const std::shared_ptr<Operator> &op) : Layer("Expression") {
  CHECK(op != nullptr && op->op_type_ == OpType::kOperatorExpression);
  ExpressionOp *expression_op = dynamic_cast<ExpressionOp *>(op.get());

  CHECK(expression_op != nullptr) << "Expression operator is empty";
  this->op_ = std::make_unique<ExpressionOp>(*expression_op);
}

void ExpressionLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                               std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  CHECK(!inputs.empty());

  const uint32_t batch_size = outputs.size();
  CHECK(batch_size != 0);

  for (uint32_t i = 0; i < batch_size; ++i) {
    CHECK(outputs.at(i) != nullptr && !outputs.at(i)->empty());
    outputs.at(i)->Fill(0.f);
  }

  CHECK(this->op_ != nullptr && this->op_->op_type_ == OpType::kOperatorExpression);
  std::stack<std::vector<std::shared_ptr<Tensor<float>>>> op_stack;
  const std::vector<std::shared_ptr<TokenNode>> &token_nodes = this->op_->Generate();
  for (const auto &token_node : token_nodes) {
    if (token_node->num_index >= 0) {
      uint32_t start_pos = token_node->num_index * batch_size;
      std::vector<std::shared_ptr<Tensor<float>>> input_token_nodes;
      for (uint32_t i = 0; i < batch_size; ++i) {
        CHECK(i + start_pos < inputs.size());
        input_token_nodes.push_back(inputs.at(i + start_pos));
      }
      op_stack.push(input_token_nodes);
    } else {
      const int32_t op = token_node->num_index;
      CHECK(op_stack.size() >= 2) << "The number of operand is less than two";
      std::vector<std::shared_ptr<Tensor<float>>> input_node1 = op_stack.top();

      CHECK(input_node1.size() == batch_size);
      op_stack.pop();

      std::vector<std::shared_ptr<Tensor<float>>> input_node2 = op_stack.top();
      CHECK(input_node2.size() == batch_size);
      op_stack.pop();

      CHECK(input_node1.size() == input_node2.size());
      std::vector<std::shared_ptr<Tensor<float>>> output_token_nodes(batch_size);
      for (uint32_t i = 0; i < batch_size; ++i) {
        if (op == -int(TokenType::TokenAdd)) {
          output_token_nodes.at(i) = ftensor::ElementAdd(input_node1.at(i), input_node2.at(i));
        } else if (op == -int(TokenType::TokenMul)) {
          output_token_nodes.at(i) = ftensor::ElementMultiply(input_node1.at(i), input_node2.at(i));
        } else {
          LOG(FATAL) << "Unknown operator type: " << op;
        }
      }
      op_stack.push(output_token_nodes);
    }
  }

  CHECK(op_stack.size() == 1);
  std::vector<std::shared_ptr<Tensor<float>>> output_node = op_stack.top();
  op_stack.pop();
  for (int i = 0; i < batch_size; ++i) {
    CHECK(outputs.at(i) != nullptr && !outputs.at(i)->empty());
    outputs.at(i) = output_node.at(i);
  }
}
}