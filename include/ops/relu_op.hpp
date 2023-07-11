//
// Created by fss on 22-12-20.
//

#ifndef KUIPER_COURSE_INCLUDE_OPS_RELU_OP_HPP_
#define KUIPER_COURSE_INCLUDE_OPS_RELU_OP_HPP_
#include "op.hpp"
namespace kuiper_infer
{
  class ReluOperator : public Operator
  {
  public:
    // override是为了保证重写的函数，与基类有相同的签名
    ~ReluOperator() override = default;
    explicit ReluOperator(float thresh);
    void set_thresh(float thresh);
    float get_thresh() const;

  private:
    float thresh_ = 0.f; // 存储阈值
  };
}
#endif // KUIPER_COURSE_INCLUDE_OPS_RELU_OP_HPP_
