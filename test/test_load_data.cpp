//
// Created by fss on 22-12-19.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "data/load_data.hpp"

TEST(test_data_load, load_csv1) {
  using namespace kuiper_infer;

  const std::string &file_path = "../tmp/data1.csv";
  std::shared_ptr<Tensor<float>> data = CSVDataLoader::LoadData(file_path, ',');
  uint32_t index = 1;
  uint32_t rows = data->rows();
  uint32_t cols = data->cols();
  ASSERT_EQ(rows, 3);
  ASSERT_EQ(cols, 6);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      ASSERT_EQ(data->at(0, r, c), index);
      index += 1;
    }
  }
}

TEST(test_data_load, load_csv_with_head1) {
  using namespace kuiper_infer;
  const std::string &file_path = "../tmp/data2.csv";
  std::vector<std::string> headers;
  std::shared_ptr<Tensor<float>> data = CSVDataLoader::LoadDataWithHeader(file_path, headers, ',');

  uint32_t index = 1;
  uint32_t rows = data->rows();
  uint32_t cols = data->cols();
  LOG(INFO) << "\n" << data;
  ASSERT_EQ(rows, 3);
  ASSERT_EQ(cols, 3);
  ASSERT_EQ(headers.size(), 3);

  ASSERT_EQ(headers.at(0), "ROW1");
  ASSERT_EQ(headers.at(1), "ROW2");
  ASSERT_EQ(headers.at(2), "ROW3");

  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      ASSERT_EQ(data->at(0, r, c), index);
      index += 1;
    }
  }

}