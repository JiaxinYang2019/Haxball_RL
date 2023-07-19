#ifndef _BASEAGENT_H_
#define _BASEAGENT_H_

#include "Eigen/Dense"

///
/// \brief The BaseAgent class
///
/// Provides an interface for agents.
/// Your agent must subclass the BaseAgent and implement the pure virtual functions.
///
class BaseAgent
{
public:

  ///
  /// \brief policy
  /// \param state the continuous state from the environment
  /// \param action the continuous action according to the policy
  ///
  /// This method gets used as feedback controller by the HaxBall Gui and performance evaluation tool.
  /// It has to convert a given continuous state into an hopefully optimal action.
  /// To be on the safe side, the method uses a const modifier such that your agent cannot get changed by accident during the execution of an action.
  /// This means that you:
  /// * cannot access a random number generator from the outside
  /// * should not define a local one because of speed
  /// * can use the policy with multi threading
  ///
  virtual void policy(const Eigen::Ref<const Eigen::VectorXd>& state,
                      Eigen::Ref<Eigen::VectorXd> action) const = 0;

  ///
  /// \brief reward
  /// \param state the continuous state from the environment
  /// \param action the continuous action executed in that state
  /// \param state_prime the continuous successor state for the (s,a) tuple
  /// \return a real valued, scalar and bounded one-step-reward r(s, a, s')
  ///
  /// This pure virtual method enforeces the signature for a reward function.
  /// The reward function computes for a one-step transition the reward (surprise).
  /// It has to be part of the base agent, such that the GUI and performance evaluation tool can have access to the collected reward over time.
  ///
  /// Use in your reward only the three available inputs to avoid defining the reward with quantities the agent cannot observe.
  /// To enforce this protection, the function uses a const modifier.
  /// Your agent can store an extra but constant instance of HaxBall to access constant information such as the goal area.
  ///
  virtual double reward(const Eigen::Ref<const Eigen::VectorXd>& state,
                        const Eigen::Ref<const Eigen::VectorXd>& action,
                        const Eigen::Ref<const Eigen::VectorXd>& state_prime) const = 0;

  ///
  /// \brief getQfactor
  /// \param state a continuous state from the environment
  /// \param action a contunuous action, which could be executed in the environment
  /// \return a Q-factor for the state action tuple
  ///
  /// This methods computers and returns for the given state action tuple the Q-factor.
  /// The main usecase for this function is to allow the gui and performance evaluation code to access the expected
  /// discounted reward for tracking the progress.
  ///
  /// For your agent, it should be better to rely on the internal representation of the Q-function, e.g.
  /// indexing a grid or evaluation features directly.
  ///
  virtual double getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state,
                            const Eigen::Ref<const Eigen::VectorXd>& action) const = 0;

  ///
  /// \brief getQfactor
  /// \param state a continuous state from the environment
  /// \return Q-factor for all available actions in the given state
  ///
  /// This methods computes for the given state the Q-factors for all actions.
  /// The main usecase for this function is to allow the gui and performance evaluation code to access the expected
  /// discounted reward for tracking the progress.
  ///
  /// For your agent it should be better to rely on the internal representation of the Q-function, e.g.
  /// extracting a whole row of your grid in one go.
  ///
  /// It makes use of the return value and thus a copy of Q such that the caller does not need to know the amount of
  /// Q-values, since a reference needs to have the correct size in advance.
  /// Therefore, this code is less performant but also significantly easier to use
  ///
  /// \overload
  ///
  virtual Eigen::VectorXd getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state) const = 0;

private:

};

#endif
