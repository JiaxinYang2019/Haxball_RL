#ifndef REWARDFUNCTIONS_H
#define REWARDFUNCTIONS_H

#include "Eigen/Dense"

namespace Reward
{
  ///
  /// \brief Distance of the player from the origin
  /// \param state Current state
  /// \param action Executed action
  /// \param state_prime Successor state of the tuple (s,a)
  /// \return Negative distance
  ///
  /// Negative Euclidean norm of player position, such that the agent should learn to steer towards the center
  ///
  /// Dense Reward / Reward Shaping
  ///
  double distance_player_origin(const Eigen::Ref<const Eigen::VectorXd>& state,
                                const Eigen::Ref<const Eigen::VectorXd>& action,
                                const Eigen::Ref<const Eigen::VectorXd>& state_prime);

  ///
  /// \brief Distance of the player from the ball
  /// \param state Current state
  /// \param action Executed action
  /// \param state_prime Successor state of the tuple (s,a)
  /// \return Negative distance
  ///
  /// Negative Euclidean norm of difference vector between player and ball position.
  /// The agent should learn to steer towards the ball, but suffers from offset since the velocity is not included
  ///
  /// Dense Reward / Reward Shaping
  ///
  double distance_player_ball_dense(const Eigen::Ref<const Eigen::VectorXd>& state,
                                    const Eigen::Ref<const Eigen::VectorXd>& action,
                                    const Eigen::Ref<const Eigen::VectorXd>& state_prime);

  ///
  /// \brief Binary indicator, whether the player is close to the ball
  /// \param state Current state
  /// \param action Executed action
  /// \param state_prime Successor state of the tuple (s,a)
  /// \return zero or one
  ///
  /// A binary indicator, whether the agent is close to the ball.
  /// Takes their radii into account.
  ///
  /// Sparse Reward
  ///
  double distance_player_ball_sparse(const Eigen::Ref<const Eigen::VectorXd>& state,
                                     const Eigen::Ref<const Eigen::VectorXd>& action,
                                     const Eigen::Ref<const Eigen::VectorXd>& state_prime);

  ///
  /// \brief Reward depending on the goal state
  /// \param state Current state
  /// \param action Executed action
  /// \param state_prime Successor state of the tuple (s,a)
  /// \return zero or one
  ///
  /// A tertiary indicator, whether the ball is in a goal area.
  /// r > 0 for enemy goal
  /// r < 0 for own goal
  /// r = 0 else
  ///
  /// Sparse Reward
  ///
  double ball_in_goal(const Eigen::Ref<const Eigen::VectorXd>& state,
                      const Eigen::Ref<const Eigen::VectorXd>& action,
                      const Eigen::Ref<const Eigen::VectorXd>& state_prime);
}

#endif // REWARDFUNCTIONS_H
