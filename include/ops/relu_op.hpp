//
// Created by fss on 22-12-20.
//

#ifndef KUIPER_COURSE_INCLUDE_OPS_RELU_OP_HPP_
#define KUIPER_COURSE_INCLUDE_OPS_RELU_OP_HPP_
#include "op.hpp"
namespace kuiper_infer {
class ReluOperator : public Operator {
 public:
  ~ReluOperator() override = default;

  explicit ReluOperator(float thresh);

  void set_thresh(float thresh);

  float get_thresh() const;

 private:
  float thresh_ = 0.f;
};
}
#endif //KUIPER_COURSE_INCLUDE_OPS_RELU_OP_HPP_
