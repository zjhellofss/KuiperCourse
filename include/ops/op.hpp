//
// Created by fss on 22-12-20.
//

#ifndef KUIPER_COURSE_INCLUDE_OPS_OP_HPP_
#define KUIPER_COURSE_INCLUDE_OPS_OP_HPP_
namespace kuiper_infer
{
  // 算子类型枚举
  enum class OpType
  {
    kOperatorUnknown = -1,
    kOperatorRelu = 0,
  };

  class Operator
  {
  public:
    // 存储算子的类型
    OpType op_type_ = OpType::kOperatorUnknown; // 不是一个具体节点 制定为unknown
    virtual ~Operator() = default;              // 指定为虚函数是为了让Operator成为一个抽象类
    explicit Operator(OpType op_type);
  };
}
#endif // KUIPER_COURSE_INCLUDE_OPS_OP_HPP_
