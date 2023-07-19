#ifndef _EVALUATIONCENTER_H_
#define _EVALUATIONCENTER_H_

#include <memory>
#include <vector>

#include "HaxBall.h"
#include "BaseAgent.h"

///
/// \brief The EvaluationCenter runs tests with your agent and tracks the progress
///
/// This class is responsible for storing the progress of your agent.
/// Set it up once and just call the evaluation regulary during your training.
/// The evaluation center computes some performance indicators and appends them to a file.
/// There is a Python script with some rudimentary plotting!
///
class EvaluationCenter
{
public:

  ///
  /// \brief Creates a new instance for evaluating agents
  /// \param agent the constant reference to your agent, i.e., the policy you want to execute
  /// \param world the shared pointer to the world in which the agent operates
  ///
  /// Sets up a new instance of the evaluation center.
  ///
  /// This constructor takes an existing agent and world and runs some tests.
  /// The world you provide mostly varies between whether or not there is an opponent.
  ///
  explicit EvaluationCenter(const BaseAgent& agent, std::shared_ptr<HaxBall> world, double gamma);

  ///
  /// \brief Creates a new instance for evaluating agents
  /// \param agent the constant reference to your agent, i.e., the policy you want to execute
  ///
  /// \overload
  ///
  explicit EvaluationCenter(const BaseAgent& agent, double gamma);

  virtual ~EvaluationCenter();

  ///
  /// \brief evaluate
  ///
  /// Calculates all performance indicators and appends them to the .csv file.
  ///
  void evaluate();

  ///
  /// \brief Runs a rollout from the given state
  /// \param start_state The start state
  /// \return the discounted return
  ///
  /// The code to run a rollout from a given start state.
  /// Resets the environment, sets the state and collects transitions.
  ///
  double rollout(const Eigen::Ref<const Eigen::VectorXd>& start_state);

private:

  ///
  /// \brief writeHeader
  ///
  /// Creates the first row with column names in the .csv file.
  /// Overwrites any existing content!
  ///
  void writeHeader() const;

  ///
  /// \brief writePropbes
  ///
  /// Creates a .csv file which contains all the sampled start states.
  ///
  void createProbes();

public:

  /// The number of start states used to compute performance values of the agent
  static const unsigned int N = 10;

  /// The length of rollouts, should be long enough to reflect gamma
  static const unsigned int TAU = 1000;

private:

  /// This HaxBall instance is used for testing an agent and to querry the rendering details
  std::shared_ptr<HaxBall> m_world;

  /// This constant reference to an agent is the agent to test. It is constant to prevent change to the internal state during testing (e.g. manipulating the random engine)
  const BaseAgent& m_agent;

  /// The list of states, which serves as probes for measuring the performance of the agent
  /// States are sampled uniformly from the state space
  std::vector<Eigen::VectorXd> m_probes;

  /// The discount factor to accumulate the returns, should be the same value as for the agent, but there is no obligation to do so
  const double m_gamma;

};

#endif // _EVALUATIONCENTER_H_

