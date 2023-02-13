//
// Created by fss on 23-1-1.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/op.hpp"
#include "ops/maxpooling_op.hpp"
#include "layer/layer.hpp"
#include "factory/layer_factory.hpp"

TEST(test_layer, forward_maxpooling1) {
  using namespace kuiper_infer;
  uint32_t stride_h = 1;
  uint32_t stride_w = 1;
  uint32_t padding_h = 0;
  uint32_t padding_w = 0;
  uint32_t pooling_h = 2;
  uint32_t pooling_w = 2;

  std::shared_ptr<Operator>
      max_op = std::make_shared<MaxPoolingOp>(pooling_h, pooling_w, stride_h, stride_w, padding_h, padding_w);
  std::shared_ptr<Layer> max_layer = LayerRegisterer::CreateLayer(max_op);
  CHECK(max_layer != nullptr);

  arma::fmat input_data = "0 1 2 ;"
                          "3 4 5 ;"
                          "6 7 8 ;";
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(2, input_data.n_rows, input_data.n_cols);
  input->at(0) = input_data;
  input->at(1) = input_data;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);

  max_layer->Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);
  const auto &output = outputs.at(0);
  LOG(INFO) << "\n" << output->data();
  ASSERT_EQ(output->rows(), 2);
  ASSERT_EQ(output->cols(), 2);

  ASSERT_EQ(output->at(0, 0, 0), 4);
  ASSERT_EQ(output->at(0, 0, 1), 5);
  ASSERT_EQ(output->at(0, 1, 0), 7);
  ASSERT_EQ(output->at(0, 1, 1), 8);

  ASSERT_EQ(output->at(1, 0, 0), 4);
  ASSERT_EQ(output->at(1, 0, 1), 5);
  ASSERT_EQ(output->at(1, 1, 0), 7);
  ASSERT_EQ(output->at(1, 1, 1), 8);
}