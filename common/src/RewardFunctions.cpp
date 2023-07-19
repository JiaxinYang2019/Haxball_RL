#include "RewardFunctions.h"

#include "HaxBall.h"

#include <iostream>

double Reward::distance_player_origin(const Eigen::Ref<const Eigen::VectorXd>& state, const Eigen::Ref<const Eigen::VectorXd>& action, const Eigen::Ref<const Eigen::VectorXd>& state_prime)
{
  Eigen::Vector2d player_pos = state.segment(0, 2);
  return -player_pos.norm();
}

double Reward::distance_player_ball_dense(const Eigen::Ref<const Eigen::VectorXd>& state, const Eigen::Ref<const Eigen::VectorXd>& action, const Eigen::Ref<const Eigen::VectorXd>& state_prime)
{
  Eigen::Vector2d player_pos = state.segment(0, 2);
  Eigen::Vector2d ball_pos = state.segment(2, 2);

  Eigen::Vector2d diff = player_pos - ball_pos ;

  double distance = diff.norm();

  // Exponential decay is also possible
  // Factors to make the function suitable for the size of the field
  // return 5.0 * std::exp(-0.5*distance);

  return -distance;
}

double Reward::distance_player_ball_sparse(const Eigen::Ref<const Eigen::VectorXd>& state, const Eigen::Ref<const Eigen::VectorXd>& action, const Eigen::Ref<const Eigen::VectorXd>& state_prime)
{
  // Static to keep memory and instance alive, const to emphasize that this should not be used actively
  static const HaxBall world;

  Eigen::Vector2d player_pos = state.segment(0, 2);
  Eigen::Vector2d ball_pos = state.segment(2, 2);

  Eigen::Vector2d diff = player_pos - ball_pos ;

  double distance = diff.norm();

  double r = world.getRadiusBall() + world.getRadiusPlayer();

  if(distance < r * 1.25)
  {
    //qDebug() << "close enough";
    return 1.0;
  }

  return 0.0;
}

double Reward::ball_in_goal(const Eigen::Ref<const Eigen::VectorXd>& state, const Eigen::Ref<const Eigen::VectorXd>& action, const Eigen::Ref<const Eigen::VectorXd>& state_prime)
{
  // Static to keep memory and instance alive, const to emphasize that this should not be used actively
  static const HaxBall world;

  // Must be the current state, because the successor state is already that with the ball at the origin
  Eigen::Vector2d ball_pos = state.segment(2, 2);

  if(world.getGoalLeft().contains(ball_pos(0), ball_pos(1)))
    return -100.0;

  if(world.getGoalRight().contains(ball_pos(0), ball_pos(1)))
    return +100.0;

  return 0.0;
}
