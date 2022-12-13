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