#include <iostream>
#include <armadillo>
int main() {
  arma::fmat in_1(32, 32, arma::fill::ones);
  arma::fmat in_2(32, 32, arma::fill::ones);

  arma::fmat out = in_1 + in_2;
  std::cout << "rows " << out.n_rows << "\n";
  std::cout << "cols " << out.n_cols << "\n";
  std::cout << "value " << out.at(0) << "\n";
  return 0;
}
