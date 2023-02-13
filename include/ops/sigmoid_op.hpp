//
// Created by fss on 22-12-29.
//

#ifndef KUIPER_COURSE_INCLUDE_OPS_SIGMOID_OP_HPP_
#define KUIPER_COURSE_INCLUDE_OPS_SIGMOID_OP_HPP_
#include "op.hpp"
#include "layer/layer.hpp"
namespace kuiper_infer {
class SigmoidOperator : public Operator {
 public:
  explicit SigmoidOperator();

};
}
#endif //KUIPER_COURSE_INCLUDE_OPS_SIGMOID_OP_HPP_
