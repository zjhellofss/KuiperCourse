//
// Created by fss on 23-1-1.
//

#include "ops/maxpooling_op.hpp"
namespace kuiper_infer {
MaxPoolingOp::MaxPoolingOp(uint32_t pooling_h, uint32_t pooling_w, uint32_t stride_h,
                           uint32_t stride_w, uint32_t padding_h, uint32_t padding_w)
    : Operator(OpType::kOperatorMaxPooling), pooling_h_(pooling_h), pooling_w_(pooling_w),
      stride_h_(stride_h), stride_w_(stride_w), padding_h_(padding_h), padding_w_(padding_w) {

}

void MaxPoolingOp::set_pooling_h(uint32_t pooling_height) {
  this->pooling_h_ = pooling_height;
}

void MaxPoolingOp::set_pooling_w(uint32_t pooling_width) {
  this->pooling_w_ = pooling_width;
}

void MaxPoolingOp::set_padding_h(uint32_t padding_height) {
  this->padding_h_ = padding_height;
}

void MaxPoolingOp::set_padding_w(uint32_t padding_width) {
  this->padding_w_ = padding_width;
}

void MaxPoolingOp::set_stride_h(uint32_t stride_height) {
  this->stride_h_ = stride_height;
}

void MaxPoolingOp::set_stride_w(uint32_t stride_width) {
  this->stride_w_ = stride_width;
}

uint32_t MaxPoolingOp::stride_width() const {
  return stride_w_;
}

uint32_t MaxPoolingOp::stride_height() const {
  return stride_h_;
}

uint32_t MaxPoolingOp::pooling_width() const {
  return pooling_w_;
}

uint32_t MaxPoolingOp::pooling_height() const {
  return pooling_h_;
}

uint32_t MaxPoolingOp::padding_width() const {
  return padding_w_;
}

uint32_t MaxPoolingOp::padding_height() const {
  return padding_h_;
}
}
