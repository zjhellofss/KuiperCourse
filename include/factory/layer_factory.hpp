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
  typedef std::shared_ptr<Layer> (*Creator)(const std::shared_ptr<Operator> &op);

  typedef std::map<OpType, Creator> CreateRegistry;

  static void RegisterCreator(OpType op_type, const Creator &creator);

  static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<Operator> &op);

  static CreateRegistry &Registry();
};

class LayerRegistererWrapper {
 public:
  LayerRegistererWrapper(OpType op_type, const LayerRegisterer::Creator &creator) {
    LayerRegisterer::RegisterCreator(op_type, creator);
  }
};

}
#endif //KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
