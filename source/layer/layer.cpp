//
// Created by fss on 22-12-20.
//
#include "layer/layer.hpp"
#include <glog/logging.h>
namespace kuiper_infer
{
  Layer::Layer(const std::string &layer_name) : layer_name_(layer_name)
  {
  }
  void Layer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                       std::vector<std::shared_ptr<Tensor<float>>> &outputs)
  {
    LOG(FATAL) << "The layer " << this->layer_name_ << " not implement yet!";
  }
}