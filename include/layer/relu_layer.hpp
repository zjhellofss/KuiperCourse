//
// Created by fss on 22-12-20.
//

#ifndef KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
#include "layer.hpp"
#include "ops/relu_op.hpp"

namespace kuiper_infer
{
  class ReluLayer : public Layer
  {
  public:
    ~ReluLayer() override = default;
    // 此处使用了算子基类的指针指向各种算子
    explicit ReluLayer(const std::shared_ptr<Operator> &op);
    // 执行relu 操作的具体函数Forwards，这个函数必须要重写
    void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                  std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
    // 根据op创建相应的layer
    static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);

  private:
    std::unique_ptr<ReluOperator> op_;
  };
}
#endif // KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
