#ifndef _QLEARNING_H_
#define _QLEARNING_H_

#include "BaseAgent.h"
#include "HaxBall.h"
#include <map>
#include <utility>
#include "Eigen/Dense"
#include <fstream>

using QTable = std::map<std::pair<double, double>, std::map<std::pair<double, double>, double>>;

class QLearning : public BaseAgent
{
private:
  const HaxBall m_world;
  std::ofstream outfile;
public:
  QLearning();
  ~QLearning();

  QTable qTable;

  double rewardValue;

  /// A linear policy
  void policy(const Eigen::Ref<const Eigen::VectorXd>& state,
              Eigen::Ref<Eigen::VectorXd> action) const override;

  /// A linear policy, which gets the parameters from the outside
  void policy(const Eigen::Ref<const Eigen::VectorXd>& state,
              const Eigen::Ref<const Eigen::VectorXd>& parameters,
              Eigen::Ref<Eigen::VectorXd> action) const;
  
  void Qpolicy(const Eigen::Ref<const Eigen::VectorXd>& state,
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

  // Function declarations
  QTable createQTable();
  std::pair<int, int> getBestAction(const QTable& qTable, const std::pair<double, double>& RoundedState) const;
  //double reward(const std::pair<int, int>& state_distance, const Eigen::Ref<const Eigen::VectorXd>& state);
  double reward(const std::pair<int, int>& state_distance, const Eigen::Ref<const Eigen::VectorXd>& state);
  void updateQTable(const Eigen::Ref<const Eigen::VectorXd>& state, QTable& qTable, const std::pair<int, int>& state_distance,const std::pair<int, int>& action, const std::pair<int, int>& nextState, double learningRate, const double& discountFactor);
  std::pair<double, double> roundedState(const Eigen::Ref<const Eigen::VectorXd>& state) const;
  float customRound(float number) const;
  void training();
  void setAction(const Eigen::Ref<const Eigen::VectorXd>& state,Eigen::Ref<Eigen::VectorXd> action, const std::pair<int, int>& Bestaction);
  void writeRewardValueToFile(double rewardValue) const;
  double calculateAverageQValue(const QTable& qTable) const;
};

#endif