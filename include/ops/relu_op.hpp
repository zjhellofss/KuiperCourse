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
  // 需要传递到reluLayer中，怎么传递？
  float thresh_ = 0.f; // 用于过滤tensor<float>值当中大于thresh的部分
  // relu存的变量只有thresh
  // stride padding kernel_size 这些是到时候convOperator需要的
  // operator起到了属性存储、变量的作用
  // operator所有子类不负责具体运算
  // 具体运算由另外一个类Layer类负责
  // y =x  , if x >=0 y = 0 if x < 0

};
}
#endif //KUIPER_COURSE_INCLUDE_OPS_RELU_OP_HPP_
