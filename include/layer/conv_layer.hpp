//
// Created by fss on 23-2-2.
//

#ifndef KUIPER_COURSE_INCLUDE_LAYER_CONV_LAYER_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_CONV_LAYER_HPP_
#include "layer.hpp"
#include "ops/conv_op.hpp"
namespace kuiper_infer {
class ConvolutionLayer : public Layer {
 public:
  explicit ConvolutionLayer(const std::shared_ptr<Operator> &op);

  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);
 private:
  std::unique_ptr<ConvolutionOp> op_;
};
}
#endif //KUIPER_COURSE_INCLUDE_LAYER_CONV_LAYER_HPP_
