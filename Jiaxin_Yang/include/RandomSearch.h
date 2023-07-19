#ifndef _RANDOMSEARCH_H_
#define _RANDOMSEARCH_H_

#include "BaseAgent.h"
#include "HaxBall.h"

#include "Eigen/Dense"

///
/// \brief The RandomSearch class
///
/// A random search agent to improve directly a linear policy.
///
class RandomSearch : public BaseAgent
{
public:
  RandomSearch();
  ~RandomSearch();

  /// A linear policy
  void policy(const Eigen::Ref<const Eigen::VectorXd>& state,
              Eigen::Ref<Eigen::VectorXd> action) const override;

  /// A linear policy, which gets the parameters from the outside
  void policy(const Eigen::Ref<const Eigen::VectorXd>& state,
              const Eigen::Ref<const Eigen::VectorXd>& parameters,
              Eigen::Ref<Eigen::VectorXd> action) const;

  /// Currently hardcoded to go to zero
  double reward(const Eigen::Ref<const Eigen::VectorXd>& s,
                const Eigen::Ref<const Eigen::VectorXd>& action,
                const Eigen::Ref<const Eigen::VectorXd>& s_prime) const override;

  /// Currently the constant Q-value "42"
  double getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state,
                    const Eigen::Ref<const Eigen::VectorXd>& action) const override;

  /// Currently the constant Q-values "42 ... 42"
  Eigen::VectorXd getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state) const override;

  ///
  /// \brief training
  ///
  /// Trains the policy by performing one iteration of the Cross Entroy Method
  ///
  void training();

private:

  /// A passive and private game instance for getting details about the game (e.g. the goal position for the reward computation)
  /// Do not use this single instance for multithreaded training, as this would mess up its internal state
  /// (That is the reason why this instance is constant)
  const HaxBall m_world;

  /// The parameters to represent a linear policy, also the mean of the Gaussian used in CEM
  Eigen::VectorXd m_parameters;

  /// The covariance matrix used to define the gaussian in Eigen
  Eigen::MatrixXd m_covariance;

public:

  /// Total number of particles for CEM
  static const unsigned int N_TOTAL;

  /// Number of particles to keep
  static const unsigned int N_KEEP;

  /// Rollout length
  static const unsigned int TAU;

  /// Discounting for rollouts
  static const double GAMMA;

};


#endif
