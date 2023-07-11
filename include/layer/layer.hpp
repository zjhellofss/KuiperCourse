//
// Created by fss on 22-12-20.
//

#ifndef KUIPER_COURSE_INCLUDE_LAYER_LAYER_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_LAYER_HPP_
#include <string>
#include "data/tensor.hpp"
namespace kuiper_infer
{
  class Layer
  {
  public:
    explicit Layer(const std::string &layer_name);
    // 算子执行接口
    virtual void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                          std::vector<std::shared_ptr<Tensor<float>>> &outputs);
    virtual ~Layer() = default;

  private:
    // 存储算子类型
    std::string layer_name_; // relu layer "relu"
  };
}
#endif // KUIPER_COURSE_INCLUDE_LAYER_LAYER_HPP_
