//
// Created by fss on 22-12-21.
//

#ifndef KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
#define KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
#include "ops/op.hpp"
#include "layer/layer.hpp"

namespace kuiper_infer
{
  class LayerRegisterer
  {
  public:
    // 这是一个函数指针，输入参数为const std::shared_ptr<Operator> &op，输出参数为std::shared_ptr<Layer>
    typedef std::shared_ptr<Layer> (*Creator)(const std::shared_ptr<Operator> &op);
    // 这是注册表类型的别名CreateRegistry
    typedef std::map<OpType, Creator> CreateRegistry;
    // 注册一下算子，将算子的信息塞到map中
    static void RegisterCreator(OpType op_type, const Creator &creator);
    // 创建对应的layer并返回
    static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<Operator> &op);
    static CreateRegistry &Registry();
  };

  class LayerRegistererWrapper // 这个是为了实现算子的注册
  {
  public:
    LayerRegistererWrapper(OpType op_type, const LayerRegisterer::Creator &creator)
    {
      LayerRegisterer::RegisterCreator(op_type, creator);
    }
  };

}
#endif // KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
