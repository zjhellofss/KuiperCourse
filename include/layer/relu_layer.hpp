//
// Created by fss on 22-12-20.
//

#ifndef KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
#include "layer.hpp"
#include "ops/relu_op.hpp"

namespace kuiper_infer {
class ReluLayer : public Layer {
 public:
  // 通过这里，把relu_op中的thresh告知给relu layer, 因为计算的时候要用到
  explicit ReluLayer(const std::shared_ptr<Operator> &op);

  // 执行relu 操作的具体函数Forwards
  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);

 private:
  // 他是存放属性用的,存放一个operator对应的属性,
  // layer在运行的时候需要其中的属性
  std::unique_ptr<ReluOperator> op_;
};
// 将relulayer(初始化方法)放入注册表的时机是什么
// 初始化方法来得到layer
}
#endif //KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
