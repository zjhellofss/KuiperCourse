//
// Created by fss on 22-12-21.
//

#include "factory/layer_factory.hpp"
#include <glog/logging.h>

namespace kuiper_infer {
void LayerRegisterer::RegisterCreator(OpType op_type, const Creator &creator) {
  CHECK(creator != nullptr) << "Layer creator is empty";
  CreateRegistry &registry = Registry();
  CHECK_EQ(registry.count(op_type), 0) << "Layer type: " << int(op_type) << " has already registered!";
  registry.insert({op_type, creator});
}

std::shared_ptr<Layer> LayerRegisterer::CreateLayer(const std::shared_ptr<Operator> &op) {
  CreateRegistry &registry = Registry();
  const OpType op_type = op->kOpType;

  LOG_IF(FATAL, registry.count(op_type) <= 0) << "Can not find the layer type: " << int(op_type);
  const auto &creator = registry.find(op_type)->second;

  LOG_IF(FATAL, !creator) << "Layer creator is empty!";
  std::shared_ptr<Layer> layer = creator(op);
  LOG_IF(FATAL, !layer) << "Layer init failed!";
  return layer;
}

LayerRegisterer::CreateRegistry &LayerRegisterer::Registry() {
  static CreateRegistry *kRegistry = new CreateRegistry();
  CHECK(kRegistry != nullptr) << "Global layer register init failed!";
  return *kRegistry;
}
}
