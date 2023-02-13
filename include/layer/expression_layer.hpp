//
// Created by fss on 23-1-22.
//

#ifndef KUIPER_COURSE_INCLUDE_LAYER_EXPRESSION_LAYER_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_EXPRESSION_LAYER_HPP_
#include "layer.hpp"
#include "ops/op.hpp"
#include "ops/expression_op.hpp"

namespace kuiper_infer {
class ExpressionLayer : public Layer {
 public:
  explicit ExpressionLayer(const std::shared_ptr<Operator> &op);
  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
 private:
  std::unique_ptr<ExpressionOp> op_;
};
}
#endif //KUIPER_COURSE_INCLUDE_LAYER_EXPRESSION_LAYER_HPP_
