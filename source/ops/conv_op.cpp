//
// Created by fss on 23-2-2.
//

#include "ops/conv_op.hpp"

namespace kuiper_infer {
bool ConvolutionOp::is_use_bias() const {
  return use_bias_;
}

void ConvolutionOp::set_use_bias(bool use_bias) {
  use_bias_ = use_bias;
}

uint32_t ConvolutionOp::groups() const {
  return groups_;
}

void ConvolutionOp::set_groups(uint32_t groups) {
  groups_ = groups;
}

uint32_t ConvolutionOp::padding_h() const {
  return padding_h_;
}

void ConvolutionOp::set_padding_h(uint32_t padding_h) {
  padding_h_ = padding_h;
}

uint32_t ConvolutionOp::padding_w() const {
  return padding_w_;
}

void ConvolutionOp::set_padding_w(uint32_t padding_w) {
  padding_w_ = padding_w;
}

uint32_t ConvolutionOp::stride_h() const {
  return stride_h_;
}

void ConvolutionOp::set_stride_h(uint32_t stride_h) {
  stride_h_ = stride_h;
}

uint32_t ConvolutionOp::stride_w() const {
  return stride_w_;
}

void ConvolutionOp::set_stride_w(uint32_t stride_w) {
  stride_w_ = stride_w;
}

void ConvolutionOp::set_weights(std::vector<std::shared_ptr<ftensor>> &weight) {
  this->weight_ = weight;
}

void ConvolutionOp::set_biases(std::vector<std::shared_ptr<ftensor>> &bias) {
  this->bias_ = bias;
}

const std::vector<std::shared_ptr<ftensor >> &ConvolutionOp::weight() const {
  return this->weight_;
}

const std::vector<std::shared_ptr<ftensor >> &ConvolutionOp::bias() const {
  return this->bias_;
}
}

