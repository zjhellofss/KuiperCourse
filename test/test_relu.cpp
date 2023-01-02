//
// Created by fss on 22-12-20.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/relu_op.hpp"
#include "layer/relu_layer.hpp"
#include "factory/layer_factory.hpp"
// op list
// conv 1
// conv 2
// relu
// sigmoid
// linear
// conv 3

//  有注册机制后的理论调用
/**
 * //无论家多少个op
 * 都只有两句C++
 * ops:[] = {conv 1,conv 2,relu , sigmod,linear,conv 3}
 * layers = []
 * for op in ops:
 *    layers.append(LayerRegisterer::CreateLayer(op))
 *    //初始化完毕
 */

// 如果没有注册机制呢?
/** 模型多少层,他多少次,这就是意义!
 *  ops:[] = {conv 1,conv 2,relu , sigmod,linear,conv 3}
 *  ConvLayer conv1 = std::make_shared(conv1_op);
 *  ConvLayer conv2 = std::make_shared(conv1_op);
 *  ReluLayer relu1 = std::make_shared(relu_op);
 *  SigmoidLayer sig = std::make_shared( sigmod_op);
 *   SigmoidLayer sig1 = std::make_shared( sigmod_op1);
     layers.append(conv1)
     layers.append(conv2)
     layers.append(relu1)
     layers.append(sig)
     layers.append(sig1)

     只是一个四层网络,resnet 几百层
     不能一个一个写!
 *
 */

// 上一节当中 没有注册机制的时候 我们是怎么做的
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
//记得切换分支!!!!!
  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}

// 有了注册机制后的框架是如何init layer
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