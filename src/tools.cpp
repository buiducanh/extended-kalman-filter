#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

const double eps = 0.000001;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd>& estimations,
    const vector<VectorXd>& ground_truth)
{
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size()
      || estimations.size() == 0) {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  for (int i = 0; i < estimations.size(); i++) {
    VectorXd resi = estimations[i] - ground_truth[i];

    resi = resi.array() * resi.array();

    rmse += resi;
  }

  rmse = rmse / estimations.size();

  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state)
{
  MatrixXd Hj(3, 4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //pre-compute a set of terms to avoid repeated calculation
  float c1 = px * px + py * py;
  float c2 = sqrt(c1);
  float c3 = (c1 * c2);

  //check division by zero
  if (fabs(c1) < eps) {
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    return Hj;
  }

  //compute the Jacobian matrix
  Hj << (px / c2), (py / c2), 0, 0,
      -(py / c1), (px / c1), 0, 0,
      py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2, py / c2;

  return Hj;
}

VectorXd Tools::StateToRadar(const VectorXd& x_state)
{
  VectorXd z(3);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  float p = sqrt(px * px + py * py);
  float phi = atan2(py, px);
  float pdot = fabs(p) < eps ? 0 : (px * vx + py * vy) / p;

  z << p, phi, pdot;

  return z;
}
