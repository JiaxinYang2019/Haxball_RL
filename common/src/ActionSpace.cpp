#include "ActionSpace.h"

#include <sstream>  // Currently only for the error message in action_map

void Action::action_map(int action, Eigen::Ref<Eigen::VectorXd> action_mapped)
{
  switch (action)
  {
  case 0:
    action_mapped << 0.0 ,0.0, 0.0; // No op
    break;
  case 1:
    action_mapped << 0.0, 0.0, 1.0; // No move but shoot
    break;
  case 2:
    action_mapped << 0.0, -1.0, 0.0; // North
    break;
  case 3:
    action_mapped << 0.0, -1.0, 1.0; // North shoot
    break;
  case 4:
    action_mapped << -1.0, -1.0, 0.0; // North West
    break;
  case 5:
    action_mapped << -1.0, -1.0, 1.0; // North West Shoot
    break;
  case 6:
    action_mapped << -1.0, 0.0, 0.0; // West
    break;
  case 7:
    action_mapped << -1.0 , 0.0, 1.0; // West Shoot
    break;
  case 8:
    action_mapped << -1.0, 1.0, 0.0; // South West
    break;
  case 9:
    action_mapped << -1.0, 1.0, 1.0 ; // South West Shoot
    break;
  case 10:
    action_mapped << 0.0, 1.0, 0.0; // South
    break;
  case 11:
    action_mapped << 0.0, 1.0, 1.0; // South Shoot
    break;
  case 12:
    action_mapped << 1.0, 1.0, 0.0; // South East
    break;
  case 13:
    action_mapped << 1.0, 1.0, 1.0; // South East Shoot
    break;
  case 14:
    action_mapped << 1.0, 0.0, 0.0; // East
    break;
  case 15:
    action_mapped << 1.0, 0.0, 1.0; // East Shoot
    break;
  case 16:
    action_mapped << 1.0, -1.0, 0.0; // North East
    break;
  case 17:
    action_mapped << 1.0, -1.0, 1.0; // North East Shoot
    break;

  default:
    std::stringstream ss;
    ss << "Invalid action: " << action;
    throw std::runtime_error(ss.str());
  }
}

int Action::action_map(const Eigen::Ref<const Eigen::VectorXd>& action)
{
  double d_cur, d_best = 1e6;
  int a_best = 0;

  Eigen::VectorXd a_cur(action.rows());

  for (int a = 0; a < 18; ++a)
  {
    action_map(a, a_cur);

    d_cur = (a_cur - action).norm();

    if(d_cur < d_best)
    {
      d_best = d_cur;
      a_best = a;
    }
  }

  return a_best;
}

