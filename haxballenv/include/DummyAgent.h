#ifndef _DUMMYAGENT_H_
#define _DUMMYAGENT_H_

#include <random>

#include "BaseAgent.h"
#include "HaxBall.h"

#include "Eigen/Dense"

///
/// \brief The DummyAgent class
///
/// A dummy agent makes the code compile, it executes a hardcoded action in all states.
///
/// This agent also contains some training functions, e.g. a proposal for structuring your code later on.
///
class DummyAgent : public BaseAgent
{
public:
  DummyAgent();
  ~DummyAgent();

  /// Currently the policy "constant right and shoot"
  void policy(const Eigen::Ref<const Eigen::VectorXd>& state,
              Eigen::Ref<Eigen::VectorXd> action) const override;


  /// Currently the constant reward signal "42"
  double reward(const Eigen::Ref<const Eigen::VectorXd>& s,
                const Eigen::Ref<const Eigen::VectorXd>& action,
                const Eigen::Ref<const Eigen::VectorXd>& s_prime) const override;

  /// Currently the constant Q-value "42"
  double getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state,
                    const Eigen::Ref<const Eigen::VectorXd>& action) const override;

  /// Currently the constant Q-values "42 ... 42"
  Eigen::VectorXd getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state) const override;

  ///
  /// \brief training Launcher for the training process
  /// \param length The length of trajectories in the HaxBall world
  /// \param trajectories How many trajectories you generate
  /// \param threads The number of threads to process all trajectories
  ///
  /// Launches training threads, nothing more in here
  ///
  void training(int length, int trajectories, int threads = 1);

private:

  ///
  /// \brief training_worker The actual place where training happens
  /// \param length The length of a trajectory in the HaxBall world
  ///
  /// Put the training code and an environment instance in here, such that each thread has
  /// its private copy of the memory.
  ///
  /// Except for the Q-function, you are perfectly thread safe
  ///
  /// For the (tabular) Q-function, race conditions do not play an important role.
  /// Bellman's equation is robust to such errors and it is unlikely that 2 threads collide in
  /// a huge Q-table.
  /// Values that change during reading will be corrected over time.
  ///
  void training_worker(int length);

private:

  /// A passive and private game instance for getting details about the game (e.g. the goal position for the reward computation)
  /// Do not use this single instance for multithreaded training, as this would mess up its internal state
  /// (That is the reason why this instance is constant)
  const HaxBall m_world;

};


#endif
