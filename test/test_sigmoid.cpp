//
// Created by fss on 22-12-29.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/op.hpp"
#include "layer/layer.hpp"
#include "layer/sigmoid_layer.hpp"
#include "factory/layer_factory.hpp"

TEST(test_layer, forward_sigmoid) {
  using namespace kuiper_infer;
  std::shared_ptr<Operator> sigmoid_op = std::make_shared<SigmoidOperator>();
  std::shared_ptr<Layer> sigmoid_layer = LayerRegisterer::CreateLayer(sigmoid_op);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f;
  input->index(1) = -2.f;
  input->index(2) = 3.f;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);
  sigmoid_layer->Forwards(inputs, outputs);

  ASSERT_EQ(outputs.size(), 1); // 测试1

  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    float a = input->index(0);
    ASSERT_EQ(outputs.at(i)->index(0), 1 / (1 + std::exp(-a)) );
    ASSERT_EQ(outputs.at(i)->index(1), 1 / (1 + std::exp(-input->index(1))));
    ASSERT_EQ(outputs.at(i)->index(2), 1 / (1 + std::exp(-input->index(2))));
  }
}