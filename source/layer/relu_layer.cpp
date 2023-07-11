//
// Created by fss on 22-12-20.
//
#include <glog/logging.h>
#include "ops/relu_op.hpp"
#include "layer/relu_layer.hpp"
#include "factory/layer_factory.hpp"

namespace kuiper_infer
{
  ReluLayer::ReluLayer(const std::shared_ptr<Operator> &op) : Layer("Relu")
  {
    CHECK(op->op_type_ == OpType::kOperatorRelu) << "Operator has a wrong type: " << int(op->op_type_);
    // dynamic_cast：用于类继承层次间的指针或引用转换。主要还是用于执行“安全的向下转型（safe downcasting）”，
    // 也即是基类对象的指针或引用转换为子类对象的指针或引用。
    ReluOperator *relu_op = dynamic_cast<ReluOperator *>(op.get());
    // 如果dynamic_cast转换失败，relu_op将会为nullptr
    CHECK(relu_op != nullptr) << "Relu operator is empty";
    // 一个op实例和一个layer 一一对应 这里relu op对一个relu layer
    this->op_ = std::make_unique<ReluOperator>(relu_op->get_thresh()); // 这里相当于是又创建了一个relu_op对象
  }
  void ReluLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                           std::vector<std::shared_ptr<Tensor<float>>> &outputs)
  {
    CHECK(this->op_ != nullptr);
    CHECK(this->op_->op_type_ == OpType::kOperatorRelu);
    // 一批x，放在vec当中，理解为batchsize数量的tensor，需要进行relu操作
    const uint32_t batch_size = inputs.size();
    for (int i = 0; i < batch_size; ++i)
    {
      CHECK(!inputs.at(i)->empty());
      const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i); // 取出批次当中的一个张量
      // 对张量中的每一个元素进行运算，进行relu运算
      // 此处使用了lambda表达式
      input_data->data().transform([&](float value)
                                   {
      // 对张量中的每一个元素进行运算
      // 从operator中得到存储的属性
      float thresh = op_->get_thresh();
      // x >= thresh return x
      if (value >= thresh) {
        return value;
      } else {
        // x<= thresh return 0.f;
        return 0.f;
      } });

      // 把结果y放在outputs中
      outputs.push_back(input_data);
    }
  }
  // 根据operator创建一个对应的layer并返回
  std::shared_ptr<Layer> ReluLayer::CreateInstance(const std::shared_ptr<Operator> &op)
  {
    std::shared_ptr<Layer> relu_layer = std::make_shared<ReluLayer>(op);
    return relu_layer;
  }

  LayerRegistererWrapper kReluLayer(OpType::kOperatorRelu, ReluLayer::CreateInstance); // 注册这个算子，这是饿汉式的单例
}