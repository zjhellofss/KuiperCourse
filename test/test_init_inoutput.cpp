//
// Created by yizhu on 2023/2/13.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "runtime/runtime_ir.hpp"

TEST(test_initinoutput, init_init_input) {
  using namespace kuiper_infer;
  const std::string &param_path = "./tmp/test.pnnx.param";
  const std::string &bin_path = "./tmp/test.pnnx.bin";
  RuntimeGraph graph(param_path, bin_path);
  graph.Init();
  const auto operators = graph.operators();
  for (const auto &operator_ : operators) {
    LOG(INFO) << "type: " << operator_->type << " name: " << operator_->name;
    const std::map<std::string, std::shared_ptr<RuntimeOperand>> &
        input_operands_map = operator_->input_operands;
    for (const auto &input_operand : input_operands_map) {
      LOG(INFO) << "operand name: " << input_operand.first << " operand shape: ";
      for (const auto &dim : input_operand.second->shapes) {
        LOG(INFO) << dim << " ";
      }
    }
  }

  LOG(INFO) << "---------------------------------------";
}