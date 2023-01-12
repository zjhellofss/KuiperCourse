//
// Created by fss on 22-12-13.
//

#include <gtest/gtest.h>
#include <armadillo>
#include <glog/logging.h>

TEST(test_first, demo1) {
  LOG(INFO) << "My first test!";
  arma::fmat in_1(32, 32, arma::fill::ones);
  ASSERT_EQ(in_1.n_cols, 32);
  ASSERT_EQ(in_1.n_rows, 32);
  ASSERT_EQ(in_1.size(), 32 * 32);
}

TEST(test_first, linear) {
  arma::fmat A = "1,2,3;"
                 "4,5,6;"
                 "7,8,9;";

  arma::fmat X = "1,1,1;"
                 "1,1,1;"
                 "1,1,1;";

  arma::fmat bias = "1,1,1;"
                    "1,1,1;"
                    "1,1,1;";

  arma::fmat output(3, 3);
  //todo 在此处插入代码，完成output = AxX + bias的运算
  // output = ?
  output = A * X + bias;

  const uint32_t cols = 3;
  for (uint32_t c = 0; c < cols; ++c) {
    float *col_ptr = output.colptr(c);
    ASSERT_EQ(*(col_ptr + 0), 7);
    ASSERT_EQ(*(col_ptr + 1), 16);
    ASSERT_EQ(*(col_ptr + 2), 25);
  }
  LOG(INFO) << "\n" <<"Result Passed!";
}
