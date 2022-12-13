#include <iostream>
#include <armadillo>

int main() {
  arma::fmat in_1(32, 32, arma::fill::ones);
  arma::fmat in_2(32, 32, arma::fill::ones);

  arma::fmat out = in_1 + in_2;
  std::cout << out << std::endl;
  return 0;
}
