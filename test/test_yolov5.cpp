//
// Created by fss on 2023/3/21.
//
#include "data/tensor.hpp"
#include "opencv2/opencv.hpp"
#include "runtime/runtime_ir.hpp"
#include "data/load_data.hpp"
#include <gtest/gtest.h>

float Letterbox(
    const cv::Mat &image,
    cv::Mat &out_image,
    const cv::Size &new_shape = cv::Size(640, 640),
    int stride = 32,
    const cv::Scalar &color = cv::Scalar(114, 114, 114),
    bool fixed_shape = false,
    bool scale_up = false) {

  cv::Size shape = image.size();
  float r = std::min(
      (float) new_shape.height / (float) shape.height, (float) new_shape.width / (float) shape.width);
  if (!scale_up) {
    r = std::min(r, 1.0f);
  }

  int new_unpad[2]{
      (int) std::round((float) shape.width * r), (int) std::round((float) shape.height * r)};

  cv::Mat tmp;
  if (shape.width != new_unpad[0] || shape.height != new_unpad[1]) {
    cv::resize(image, tmp, cv::Size(new_unpad[0], new_unpad[1]));
  } else {
    tmp = image.clone();
  }

  float dw = new_shape.width - new_unpad[0];
  float dh = new_shape.height - new_unpad[1];

  if (!fixed_shape) {
    dw = (float) ((int) dw % stride);
    dh = (float) ((int) dh % stride);
  }

  dw /= 2.0f;
  dh /= 2.0f;

  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));
  cv::copyMakeBorder(tmp, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);

  return 1.0f / r;
}

kuiper_infer::sftensor PreProcessImage(const cv::Mat &image, const int32_t input_h, const int32_t input_w) {
  assert(!image.empty());
  using namespace kuiper_infer;
  const int32_t input_c = 3;

  int stride = 32;
  cv::Mat out_image;
  Letterbox(image, out_image, {input_h, input_w}, stride, {114, 114, 114},
            true);

  cv::Mat rgb_image;
  cv::cvtColor(out_image, rgb_image, cv::COLOR_BGR2RGB);

  cv::Mat normalize_image;
  rgb_image.convertTo(normalize_image, CV_32FC3, 1. / 255.);

  std::vector<cv::Mat> split_images;
  cv::split(normalize_image, split_images);
  assert(split_images.size() == input_c);

  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(input_c, input_h, input_w);
  input->Fill(0.f);

  int index = 0;
  int offset = 0;
  for (const auto &split_image : split_images) {
    assert(split_image.total() == input_w * input_h);
    const cv::Mat &split_image_t = split_image.t();
    memcpy(input->slice(index).memptr(), split_image_t.data,
           sizeof(float) * split_image.total());
    index += 1;
    offset += split_image.total();
  }
  return input;
}

struct Detection {
  cv::Rect box;
  float conf = 0.f;
  int class_id = -1;
};

template<typename T>
T clip(const T &n, const T &lower, const T &upper) {
  return std::max(lower, std::min(n, upper));
}

void ScaleCoords(const cv::Size &img_shape, cv::Rect &coords, const cv::Size &img_origin_shape) {
  float gain = std::min((float) img_shape.height / (float) img_origin_shape.height,
                        (float) img_shape.width / (float) img_origin_shape.width);

  int pad[2] = {(int) (((float) img_shape.width - (float) img_origin_shape.width * gain) / 2.0f),
                (int) (((float) img_shape.height - (float) img_origin_shape.height * gain) / 2.0f)};

  coords.x = (int) std::round(((float) (coords.x - pad[0]) / gain));
  coords.y = (int) std::round(((float) (coords.y - pad[1]) / gain));

  coords.width = (int) std::round(((float) coords.width / gain));
  coords.height = (int) std::round(((float) coords.height / gain));

  coords.x = clip(coords.x, 0, img_origin_shape.width);
  coords.y = clip(coords.y, 0, img_origin_shape.height);
  coords.width = clip(coords.width, 0, img_origin_shape.width);
  coords.height = clip(coords.height, 0, img_origin_shape.height);
}

void YoloDemo(const std::vector<std::string> &image_paths,
              const std::string &param_path,
              const std::string &bin_path,
              const uint32_t batch_size, const float conf_thresh = 0.25f,
              const float iou_thresh = 0.25f) {
  using namespace kuiper_infer;
  const int32_t input_h = 640;
  const int32_t input_w = 640;

  assert(batch_size == image_paths.size());
  std::vector<sftensor> inputs;
  for (uint32_t i = 0; i < batch_size; ++i) {
    const auto &input_image = cv::imread(image_paths.at(i));
    sftensor input = PreProcessImage(input_image, input_h, input_w);
    assert(input->rows() == 640);
    assert(input->cols() == 640);
    inputs.push_back(input);
  }

  RuntimeGraph graph(param_path, bin_path);
  graph.Build("pnnx_input_0", "pnnx_output_0");

  std::vector<std::shared_ptr<Tensor<float>>> outputs =
      graph.Forward(inputs, true);

  assert(outputs.size() == inputs.size());
  assert(outputs.size() == batch_size);

  for (int i = 0; i < outputs.size(); ++i) {
    const auto &image = cv::imread(image_paths.at(i));
    const int32_t origin_input_h = image.size().height;
    const int32_t origin_input_w = image.size().width;

    const auto &output = outputs.at(i);
    assert(!output->empty());
    const auto &shapes = output->shapes();
    assert(shapes.size() == 3);

    const uint32_t elements = shapes.at(1);
    const uint32_t num_info = shapes.at(2);
    std::vector<Detection> detections;

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> class_ids;

    const uint32_t b = 0;
    for (uint32_t e = 0; e < elements; ++e) {
      float cls_conf = output->at(b, e, 4);
      if (cls_conf >= conf_thresh) {
        int center_x = (int) (output->at(b, e, 0));
        int center_y = (int) (output->at(b, e, 1));
        int width = (int) (output->at(b, e, 2));
        int height = (int) (output->at(b, e, 3));
        int left = center_x - width / 2;
        int top = center_y - height / 2;

        int best_class_id = -1;
        float best_conf = -1.f;
        for (uint32_t j = 5; j < num_info; ++j) {
          if (output->at(b, e, j) > best_conf) {
            best_conf = output->at(b, e, j);
            best_class_id = int(j - 5);
          }
        }

        boxes.emplace_back(left, top, width, height);
        confs.emplace_back(best_conf * cls_conf);
        class_ids.emplace_back(best_class_id);
      }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, conf_thresh, iou_thresh, indices);

    for (int idx : indices) {
      Detection det;
      det.box = cv::Rect(boxes[idx]);
      ScaleCoords(cv::Size{input_w, input_h}, det.box,
                  cv::Size{origin_input_w, origin_input_h});

      det.conf = confs[idx];
      det.class_id = class_ids[idx];
      detections.emplace_back(det);
    }

    int font_face = cv::FONT_HERSHEY_COMPLEX;
    double font_scale = 2;

    for (const auto &detection : detections) {
      cv::rectangle(image, detection.box, cv::Scalar(255, 255, 255), 4);
      cv::putText(image, std::to_string(detection.class_id),
                  cv::Point(detection.box.x, detection.box.y), font_face,
                  font_scale, cv::Scalar(255, 255, 0), 4);
    }
    cv::imwrite(std::string("output") + std::to_string(i) + ".jpg", image);
  }
}

TEST(test_net, forward_yolo1) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/demo/yolov5n_small.pnnx.param",
                     "tmp/demo/yolov5n_small.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  const uint32_t batch_size = 4;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 320, 320);
    input->Fill(127.f);
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
  for (int i = 0; i < batch_size; ++i) {
    std::string file_path = "tmp/" + std::to_string(i + 1) + ".csv";
    const auto &output1 = CSVDataLoader::LoadData(file_path);
    const auto &output2 = outputs.at(i);

    ASSERT_EQ(output1.size(), output2->size());
    for (int r = 0; r < output1.n_rows; ++r) {
      for (int c = 0; c < output1.n_cols; ++c) {
        ASSERT_LE(std::abs(output1.at(r, c) - output2->at(0, r, c)), 0.05) << " row: " << r << " col: " << c;
      }
    }
  }
}

TEST(test_model, yolo_demo) {
  const uint32_t batch_size = 1;
  const std::vector<std::string> image_paths{"./tmp/bus.jpg"};
  const std::string &param_path = "tmp/yolov5s.pnnx.param";
  const std::string &bin_path = "tmp/yolov5s.pnnx.bin";

  YoloDemo(image_paths, param_path, bin_path, batch_size);
}