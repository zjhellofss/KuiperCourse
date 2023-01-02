//
// Created by fss on 22-12-21.
//

#include "factory/layer_factory.hpp"
#include <glog/logging.h>

namespace kuiper_infer {
// OpType::kOperatorRelu 就是刚才还说的OpType
// ReluLayer::CreateInstance就是一个函数指针,用来初始化层的方法

// 单例设计编程模式
// 全局当中有且只有一个变量
// 任意次和任意一方去调用都会得到这个唯一的变量
// 这里的唯一变量是全局的注册表 存的时候是这个,取得时候也需要是这个
// typedef std::map<OpType, Creator> CreateRegistry
// 全局当中有且只有一个 CreateRegistry  的实例
// 什么方法来控制这个变量唯一呢
void LayerRegisterer::RegisterCreator(OpType op_type, const Creator &creator) {
  CHECK(creator != nullptr) << "Layer creator is empty";
  CreateRegistry &registry = Registry(); //实现单例的关键
  // 根据operator type
  CHECK_EQ(registry.count(op_type), 0) << "Layer type: " << int(op_type) << " has already registered!";
 //  ReluLayer::CreateInstance 没有被注册过,就塞入到注册表当中
  registry.insert({op_type, creator});
}

std::shared_ptr<Layer> LayerRegisterer::CreateLayer(const std::shared_ptr<Operator> &op) {
  CreateRegistry &registry = Registry();
  const OpType op_type = op->op_type_;

  LOG_IF(FATAL, registry.count(op_type) <= 0) << "Can not find the layer type: " << int(op_type);
  // 根据传入的op_type(relu type)得到CreateInstance creator
  const auto &creator = registry.find(op_type)->second;

  LOG_IF(FATAL, !creator) << "Layer creator is empty!";
  std::shared_ptr<Layer> layer = creator(op);
  LOG_IF(FATAL, !layer) << "Layer init failed!";
  return layer;
}
//注册的是一个初始化方法
//得到的需要是根据初始化方法初始的layer
LayerRegisterer::CreateRegistry &LayerRegisterer::Registry() {
  // static CreateRegistry *kRegistry = new CreateRegistry();
  // 这个其实只会被初始化一次
  // 简单来说第一次,调用的时候 new CreateRegistry 存放到一个kRegistry (static)
  // 后续调用的时候,只会返回kRegistry (static)
  // C++ 特性

  // C++程序员高频面试点
  static  CreateRegistry *kRegistry = new CreateRegistry();
  // 没有static 那就是调用一次初始化一次
  // 不构成单例
  CHECK(kRegistry != nullptr) << "Global layer register init failed!";
  return *kRegistry; // 返回了这个注册表
}
}
