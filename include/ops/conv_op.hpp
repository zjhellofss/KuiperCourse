//
// Created by fss on 23-2-2.
//

#ifndef KUIPER_COURSE_SOURCE_OPS_CONV_OP_HPP_
#define KUIPER_COURSE_SOURCE_OPS_CONV_OP_HPP_
#include "op.hpp"
#include <cstdint>
#include <vector>
#include "data/tensor.hpp"

namespace kuiper_infer {
class ConvolutionOp : public Operator {
 public:
  explicit ConvolutionOp(bool use_bias,
                         uint32_t groups,
                         uint32_t stride_h,
                         uint32_t stride_w,
                         uint32_t padding_h,
                         uint32_t padding_w)
      : Operator(OpType::kOperatorConvolution),
        use_bias_(use_bias), groups_(groups),
        stride_h_(stride_h), stride_w_(stride_w),
        padding_h_(padding_h), padding_w_(padding_w) {

  }
  void set_weights(std::vector<std::shared_ptr<ftensor>> &weight);

  void set_biases(std::vector<std::shared_ptr<ftensor>> &bias);

  const std::vector<std::shared_ptr<ftensor >> &weight() const;

  const std::vector<std::shared_ptr<ftensor >> &bias() const;

  bool is_use_bias() const;

  void set_use_bias(bool use_bias);

  uint32_t groups() const;

  void set_groups(uint32_t groups);

  uint32_t padding_h() const;

  void set_padding_h(uint32_t padding_h);

  uint32_t padding_w() const;

  void set_padding_w(uint32_t padding_w);

  uint32_t stride_h() const;

  void set_stride_h(uint32_t stride_h);

  uint32_t stride_w() const;

  void set_stride_w(uint32_t stride_w);
 private:
  bool use_bias_ = false;
  uint32_t groups_ = 1;
  uint32_t padding_h_ = 0;
  uint32_t padding_w_ = 0;
  uint32_t stride_h_ = 1;
  uint32_t stride_w_ = 1;
  std::vector<std::shared_ptr<ftensor>> weight_;
  std::vector<std::shared_ptr<ftensor >> bias_;
};
}
#endif //KUIPER_COURSE_SOURCE_OPS_CONV_OP_HPP_
