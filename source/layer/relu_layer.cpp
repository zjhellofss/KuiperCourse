//
// Created by fss on 22-12-20.
//
#include <glog/logging.h>
#include "ops/relu_op.hpp"
#include "layer/relu_layer.hpp"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
ReluLayer::ReluLayer(const std::shared_ptr<Operator> &op) : Layer("Relu") {
  CHECK(op->op_type_ == OpType::kOperatorRelu) << "Operator has a wrong type: " << int(op->op_type_);
  // dynamic_cast是什么意思？ 就是判断一下op指针是不是指向一个relu_op类的指针
  // 这边的op不是ReluOperator类型的指针，就报错
  // 我们这里只接受ReluOperator类型的指针
  // 父类指针必须指向子类ReluOperator类型的指针
  // 为什么不讲构造函数设置为const std::shared_ptr<ReluOperator> &op？
  // 为了接口统一，具体下节会说到
  ReluOperator *relu_op = dynamic_cast<ReluOperator *>(op.get());

  CHECK(relu_op != nullptr) << "Relu operator is empty";
  // 一个op实例和一个layer 一一对应 这里relu op对一个relu layer
  // 对应关系
  this->op_ = std::make_unique<ReluOperator>(relu_op->get_thresh());
}

void ReluLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                         std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  // relu 操作在哪里，这里！
  // 我需要该节点信息的时候 直接这么做
  // 实行了属性存储和运算过程的分离！！！！！！！！！！！！！！！！！！！！！！！！
  //x就是inputs y = outputs
  CHECK(this->op_ != nullptr);
  CHECK(this->op_->op_type_ == OpType::kOperatorRelu);

  const uint32_t batch_size = inputs.size(); //一批x，放在vec当中，理解为batchsize数量的tensor，需要进行relu操作
  for (int i = 0; i < batch_size; ++i) {

    CHECK(!inputs.at(i)->empty());
    const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i); //取出批次当中的一个张量

    //对张量中的每一个元素进行运算，进行relu运算
    input_data->data().transform([&](float value) {
      // 对张良中的没一个元素进行运算
      // 从operator中得到存储的属性
      float thresh = op_->get_thresh();
      //x >= thresh
      if (value >= thresh) {
        return value; // return x
      } else {
        // x<= thresh return 0.f;
        return 0.f;
      }
    });

    // 把结果y放在outputs中
    outputs.push_back(input_data);
  }
}

std::shared_ptr<Layer> ReluLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
  std::shared_ptr<Layer> relu_layer = std::make_shared<ReluLayer>(op);
  return relu_layer;
}

LayerRegistererWrapper kReluLayer(OpType::kOperatorRelu, ReluLayer::CreateInstance);
}