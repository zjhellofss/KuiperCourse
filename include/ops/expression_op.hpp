//
// Created by fss on 23-1-22.
//

#ifndef KUIPER_COURSE_INCLUDE_OPS_EXPRESSION_OP_HPP_
#define KUIPER_COURSE_INCLUDE_OPS_EXPRESSION_OP_HPP_

#include <vector>
#include <string>
#include <memory>
#include "op.hpp"
#include "parser/parse_expression.hpp"

namespace kuiper_infer {
class ExpressionOp : public Operator {
 public:
  explicit ExpressionOp(const std::string &expr);
  std::vector<std::shared_ptr<TokenNode>> Generate();

 private:
  std::shared_ptr<ExpressionParser> parser_;
  std::vector<std::shared_ptr<TokenNode>> nodes_;
  std::string expr_;
};
}
#endif //KUIPER_COURSE_INCLUDE_OPS_EXPRESSION_OP_HPP_
