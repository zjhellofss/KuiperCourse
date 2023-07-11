//
// Created by fss on 22-12-20.
//
#include "ops/relu_op.hpp"
namespace kuiper_infer
{

  ReluOperator::ReluOperator(float thresh) : thresh_(thresh), Operator(OpType::kOperatorRelu)
  {
  }
  void ReluOperator::set_thresh(float thresh)
  {
    this->thresh_ = thresh;
  }
  float ReluOperator::get_thresh() const
  {
    return thresh_;
  }
}