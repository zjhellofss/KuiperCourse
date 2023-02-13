//
// Created by fss on 23-1-7.
//

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/runtime_ir.hpp"

TEST(test_runtime, runtime1) {
  using namespace kuiper_infer;
  const std::string &param_path = "./tmp/test.pnnx.param";
  const std::string &bin_path = "./tmp/test.pnnx.bin";
  RuntimeGraph graph(param_path, bin_path);
  graph.Init();
  const auto operators = graph.operators();
  for (const auto &operator_ : operators) {
    LOG(INFO) << "type: " << operator_->type << " name: " << operator_->name;
  }
}
