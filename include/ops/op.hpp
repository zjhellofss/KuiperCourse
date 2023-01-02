//
// Created by fss on 22-12-20.
//

#ifndef KUIPER_COURSE_INCLUDE_OPS_OP_HPP_
#define KUIPER_COURSE_INCLUDE_OPS_OP_HPP_
namespace kuiper_infer {
enum class OpType {
  kOperatorUnknown = -1,
  kOperatorRelu = 0,
  kOperatorSigmoid = 1,
  kOperatorMaxPooling = 2,
};

class Operator {
 public:
  OpType op_type_ = OpType::kOperatorUnknown; //不是一个具体节点 制定为unknown

  virtual ~Operator() = default; //

  explicit Operator(OpType op_type);
  explicit Operator() = default;
};

}
#endif //KUIPER_COURSE_INCLUDE_OPS_OP_HPP_
