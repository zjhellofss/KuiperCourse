//
// Created by fss on 22-12-20.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/relu_op.hpp"
#include "layer/relu_layer.hpp"
#include "factory/layer_factory.hpp"

TEST(test_layer, forward_relu1) {
  using namespace kuiper_infer;
  float thresh = 0.f;
  // 初始化一个relu operator 并设置属性
  std::shared_ptr<Operator> relu_op = std::make_shared<ReluOperator>(thresh);

  // 有三个值的一个tensor<float>
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f; //output对应的应该是0
  input->index(1) = -2.f; //output对应的应该是0
  input->index(2) = 3.f; //output对应的应该是3
  // 主要第一个算子，经典又简单，我们这里开始！

  std::vector<std::shared_ptr<Tensor<float>>> inputs; //作为一个批次去处理

  std::vector<std::shared_ptr<Tensor<float>>> outputs; //放结果
  inputs.push_back(input);
  ReluLayer layer(relu_op);
  // 因为是4.1 所以没有作业 4.2才有
// 一个批次是1
  layer.Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);

  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}

TEST(test_layer, forward_relu2) {
  using namespace kuiper_infer;
  float thresh = 0.f;
  std::shared_ptr<Operator> relu_op = std::make_shared<ReluOperator>(thresh);
  std::shared_ptr<Layer> relu_layer = LayerRegisterer::CreateLayer(relu_op);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f;
  input->index(1) = -2.f;
  input->index(2) = 3.f;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);
  relu_layer->Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}