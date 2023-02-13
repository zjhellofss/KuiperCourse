//
// Created by fss on 23-1-1.
//

#ifndef KUIPER_COURSE_INCLUDE_OPS_MAXPOOLING_OP_HPP_
#define KUIPER_COURSE_INCLUDE_OPS_MAXPOOLING_OP_HPP_
#include <cstdint>
#include "op.hpp"

namespace kuiper_infer {
class MaxPoolingOp : public Operator {
 public:
  explicit MaxPoolingOp(uint32_t pooling_h, uint32_t pooling_w, uint32_t stride_h,
                        uint32_t stride_w, uint32_t padding_h, uint32_t padding_w);

  void set_pooling_h(uint32_t pooling_height);
  void set_pooling_w(uint32_t pooling_width);

  void set_stride_w(uint32_t stride_width);
  void set_stride_h(uint32_t stride_height);

  void set_padding_h(uint32_t padding_height);
  void set_padding_w(uint32_t padding_width);

  uint32_t padding_height() const;
  uint32_t padding_width() const;

  uint32_t stride_width() const;
  uint32_t stride_height() const;

  uint32_t pooling_height() const;
  uint32_t pooling_width() const;
 private:
  uint32_t pooling_h_; // 池化核高度大小
  uint32_t pooling_w_; // 池化核宽度大小
  uint32_t stride_h_;  // 高度上的步长
  uint32_t stride_w_;  // 宽度上的步长
  uint32_t padding_h_; // 高度上的填充
  uint32_t padding_w_; // 宽度上的填充
};
}
#endif //KUIPER_COURSE_INCLUDE_OPS_MAXPOOLING_OP_HPP_
