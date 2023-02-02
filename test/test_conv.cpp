//
// Created by fss on 23-2-2.
//

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "ops/op.hpp"
#include "layer/conv_layer.hpp"

// 单卷积单通道
TEST(test_layer, conv1) {
  using namespace kuiper_infer;
  LOG(INFO) << "My convolution test!";
  ConvolutionOp *conv_op = new ConvolutionOp(false, 1, 1, 1, 0, 0);
  // 单个卷积核的情况
  std::vector<float> values;
  for (int i = 0; i < 3; ++i) {
    values.push_back(float(i + 1));
    values.push_back(float(i + 1));
    values.push_back(float(i + 1));
  }
  std::shared_ptr<ftensor> weight1 = std::make_shared<ftensor>(1, 3, 3);
  weight1->Fill(values);
  LOG(INFO) << "weight:";
  weight1->Show();
  // 设置权重
  std::vector<sftensor> weights;
  weights.push_back(weight1);
  conv_op->set_weights(weights);
  std::shared_ptr<Operator> op = std::shared_ptr<ConvolutionOp>(conv_op);

  std::vector<std::shared_ptr<ftensor >> inputs;
  arma::fmat input_data = "1,2,3,4;"
                          "5,6,7,8;"
                          "7,8,9,10;"
                          "11,12,13,14";
  std::shared_ptr<ftensor> input = std::make_shared<ftensor>(1, 4, 4);
  input->at(0) = input_data;
  LOG(INFO) << "input:";
  input->Show();
  // 权重数据和输入数据准备完毕
  inputs.push_back(input);
  ConvolutionLayer layer(op);
  std::vector<std::shared_ptr<ftensor >> outputs(1);

  layer.Forwards(inputs, outputs);
  LOG(INFO) << "result: ";
  for (int i = 0; i < outputs.size(); ++i) {
    outputs.at(i)->Show();
  }
}

// 多卷积多通道
TEST(test_layer, conv2) {
  using namespace kuiper_infer;
  LOG(INFO) << "My convolution test!";
  ConvolutionOp *conv_op = new ConvolutionOp(false, 1, 1, 1, 0, 0);
  // 单个卷积核的情况
  std::vector<float> values;

  arma::fmat weight_data = "1 ,1, 1 ;"
                           "2 ,2, 2;"
                           "3 ,3, 3;";
  // 初始化三个卷积核
  std::shared_ptr<ftensor> weight1 = std::make_shared<ftensor>(3, 3, 3);
  weight1->at(0) = weight_data;
  weight1->at(1) = weight_data;
  weight1->at(2) = weight_data;

  std::shared_ptr<ftensor> weight2 = weight1->Clone();
  std::shared_ptr<ftensor> weight3 = weight1->Clone();

  LOG(INFO) << "weight:";
  weight1->Show();
  // 设置权重
  std::vector<sftensor> weights;
  weights.push_back(weight1);
  weights.push_back(weight2);
  weights.push_back(weight3);

  conv_op->set_weights(weights);
  std::shared_ptr<Operator> op = std::shared_ptr<ConvolutionOp>(conv_op);

  std::vector<std::shared_ptr<ftensor >> inputs;
  arma::fmat input_data = "1,2,3,4;"
                          "5,6,7,8;"
                          "7,8,9,10;"
                          "11,12,13,14";
  std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 4, 4);
  input->at(0) = input_data;
  input->at(1) = input_data;
  input->at(2) = input_data;

  LOG(INFO) << "input:";
  input->Show();
  // 权重数据和输入数据准备完毕
  inputs.push_back(input);
  ConvolutionLayer layer(op);
  std::vector<std::shared_ptr<ftensor >> outputs(1);

  layer.Forwards(inputs, outputs);
  LOG(INFO) << "result: ";
  for (int i = 0; i < outputs.size(); ++i) {
    outputs.at(i)->Show();
  }
}