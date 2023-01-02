//
// Created by fss on 22-12-21.
//

#ifndef KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
#define KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
#include "ops/op.hpp"
#include "layer/layer.hpp"

namespace kuiper_infer {
class LayerRegisterer {
 public:
  // 返回值为std::shared_ptr<Layer> 然后参数为const std::shared_ptr<Operator> &op的一类函数
  // 只要返回值和参数的类型\个数都满足 creator就可以指向对应的函数
  // 对应函数就是创建层layer的一个具体方法
  typedef std::shared_ptr<Layer> (*Creator)(const std::shared_ptr<Operator> &op);

  typedef std::map<OpType, Creator> CreateRegistry;

  static void RegisterCreator(OpType op_type, const Creator &creator);

  static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<Operator> &op);

  static CreateRegistry &Registry();
};

// relulayer-->kReluLayer(LayerRegistererWrapper)--> LayerRegisterer::RegisterCreator 调用链
//
class LayerRegistererWrapper {
 public:
  LayerRegistererWrapper(OpType op_type, const LayerRegisterer::Creator &creator) {
    // 定义之后调用的
    // RegisterCreator
    LayerRegisterer::RegisterCreator(op_type, creator);
  }
};

}
#endif //KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
