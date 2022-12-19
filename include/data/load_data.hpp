//
// Created by fss on 22-12-19.
//

#ifndef KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_
#define KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_
#include <armadillo>
#include "data/tensor.hpp"

namespace kuiper_infer {
class CSVDataLoader {
 public:
  static std::shared_ptr<Tensor<float >> LoadData(const std::string &file_path, char split_char = ',');

  static std::shared_ptr<Tensor<float >> LoadDataWithHeader(const std::string &file_path,
                                       std::vector<std::string> &headers, char split_char = ',');

 private:
  static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file, char split_char);
};
}
#endif //KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_
