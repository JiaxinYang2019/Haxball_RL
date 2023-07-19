#include "QLearning.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>

#include <QApplication>
#include <QDebug>
#include <Eigen/Dense>

#include "HaxBall.h"
#include "HaxBallGui.h"

#include "RandomSearch.h"
#include "DummyAgent.h"

QLearning::QLearning() : qTable(createQTable())
{
  //std::ifstream file("rewards.csv");
  //std::ifstream file("qtable.csv");
  //std::ifstream file("goal.csv");
  //outfile.open("rewards.csv");
  //outfile.open("qtable.csv");
  //outfile.open("goal.csv");
}

QLearning::~QLearning()
{
  outfile.close();
}

// Function to create and initialize the Q-table
QTable QLearning::createQTable()
{
  QTable qTable;
  // Initialize all Q-values to 0
  for (int x1 = -8; x1 <= 8; x1+= 1) {
    for (int y1 = -4; y1 <= 4; y1+= 1) {
      std::pair<int, int> state_distance = std::make_pair(x1, y1);
      for (int x2 = -1; x2 <= 1; x2+= 1) {
        for (int y2 = -1; y2 <= 1; y2+= 1) {
          std::pair<int, int> action = std::make_pair(x2, y2);
          qTable[state_distance][action] = 0.0; // Initialize Q-value to 0
        }
      }
    }
  }  
  return qTable;  
}

double QLearning::calculateAverageQValue(const QTable& qTable) const
{
  double sumQValues = 0.0;
  int numValues = 0;

  for (const auto& stateAction : qTable)
  {
    const auto& actionQValues = stateAction.second;

    for (const auto& actionQValue : actionQValues)
    {
      sumQValues += actionQValue.second;
      numValues++;
    }
  }

  double averageQValue = (numValues > 0) ? (sumQValues / numValues) : 0.0;
  return sumQValues;
}

std::pair<int, int> QLearning::getBestAction(const QTable& qTable, const std::pair<double, double>& RoundedState) const
{
  // Find the action with the highest Q-value for the given state
  double maxQValue = std::numeric_limits<double>::lowest();
  std::pair<int, int> bestAction;
  // Iterate over the QTable using iterators
  for (const auto& stateAction : qTable)
  {
    const auto& state = stateAction.first;
    const auto& actionQValues = stateAction.second;
    if (state == RoundedState)
    {
      // Iterate over the action Q-values for the current state
      for (const auto& actionQValue : actionQValues)
      {
        if (actionQValue.second > maxQValue)
        {
          maxQValue = actionQValue.second;
          bestAction = actionQValue.first;
        }
      }
      break; // Found the desired state, no need to continue searching
    }
  }
  return bestAction; 
}

// std::pair<int, int> QLearning::getBestAction(const QTable& qTable, const std::pair<int, int>& RoundedState) const
// {
//   // Find the action with the highest Q-value for the given state
//   double maxQValue = std::numeric_limits<double>::lowest();
//   std::pair<int, int> bestAction;
//   for (const auto& actionQValue : qTable[RoundedState])
//   {
//     if (actionQValue.second > maxQValue)
//     {
//       maxQValue = actionQValue.second;
//       bestAction = actionQValue.first;
//     }
//   }
//   return bestAction; 
// }

double QLearning::reward(const std::pair<int, int>& state_distance, const Eigen::Ref<const Eigen::VectorXd>& state)
{
  // if (state[2] >= 3.9 && state[3] >= -0.7 && state[3] <= 0.7) {
  //   double x = state_distance.first;
  //   double y = state_distance.second;
  //   double norm = std::sqrt(x * x + y * y);
  //   return -norm + 100;
  // } 
  // else if (state[2] <= -3.98 && state[3] >= -0.7 && state[3] <= 0.7) {
  //   double x = state_distance.first;
  //   double y = state_distance.second;
  //   double norm = std::sqrt(x * x + y * y);
  //   return -norm - 0.1; } 
  //  else {
  //   double x = state_distance.first;
  //   double y = state_distance.second;
  //   double norm = std::sqrt(x * x + y * y);
  //   return -norm;
  // }
  double x = state_distance.first;
  double y = state_distance.second;
  double norm = std::sqrt(x * x + y * y);
  if (state[2] >= 3.95 && state[3] >= -0.7 && state[3] <= 0.7) {
    return (-norm + 20);
 } 
  else if (state[2] <= -3.95 && state[3] >= -0.4 && state[3] <= 0.4) {
    return (-norm - 20);
  } 
     else {
        return -norm;
  }
  
}

void QLearning::updateQTable(const Eigen::Ref<const Eigen::VectorXd>& state, QTable& qTable, const std::pair<int, int>& state_distance, const std::pair<int, int>& action, const std::pair<int, int>& nextState, double learningRate, const double& discountFactor)
{ // Retrieve the current Q-value for the state-action pair
  double currentQValue = qTable[state_distance][action];

  // Find the best action for the next state
  std::pair<int, int> bestNextAction = getBestAction(qTable, nextState);

  // Retrieve the Q-value of the best action in the next state
  double bestNextQValue = qTable[nextState][bestNextAction];

  // Calculate the reward (you can use the reward function here)
  double rewardvalue = reward(state_distance, state);

  // Save the reward value to the csv file
  /***TO DO***/
  //record rewardvalue into outfile in 1000 timestep//
  /***TO DO***/

  //outfile << rewardvalue << std::endl;

  // Apply the Q-learning update rule
  double newQValue = currentQValue + learningRate * (rewardvalue + discountFactor * bestNextQValue - currentQValue);

  // Update the Q-value in the Q-table
  qTable[state_distance][action] = newQValue;
}

////////////////////// change here to distance //////////////////////
// std::pair<double, double> QLearning::roundedState(const Eigen::Ref<const Eigen::VectorXd>& state) const
// {
//     double roundedNumber0 = std::round(state[0] - state[2]);
//     double roundedNumber1 = std::round(state[1] - state[3]);
//     double roundedState0 = std::max(-8.0, std::min(roundedNumber0, 8.0));
//     double roundedState1 = std::max(-4.0, std::min(roundedNumber1, 4.0));

//     // Adjust the rounded state values to the desired resolution
//     roundedState0 = std::round(roundedState0 * 2.0) / 2.0;
//     roundedState1 = std::round(roundedState1 * 2.0) / 2.0;

//     return std::make_pair(roundedState0, roundedState1);
// }
std::pair<double, double> QLearning::roundedState(const Eigen::Ref<const Eigen::VectorXd>& state) const
{
    int roundedNumber0 = static_cast<int>(std::round(state[0] - state[2])); //distance x --> check the meaning of this
    int roundedNumber1 = static_cast<int>(std::round(state[1] - state[3])); //distance y --> check the meaning of this
    int roundedState0 = std::max(-8, std::min(roundedNumber0, 8)); //  check the meaning ot this again
    int roundedState1 = std::max(-4, std::min(roundedNumber1, 4)); // 
    return std::make_pair(roundedState0, roundedState1);
}
/***TO DO***/
/*change the resoulition to 0.5*/
/***TO DO***/

void QLearning::setAction(const Eigen::Ref<const Eigen::VectorXd>& state,
                          Eigen::Ref<Eigen::VectorXd> action, 
                          const std::pair<int, int>& Bestaction)

{
  const unsigned int dim = m_world.getStateDimension();
  action << Bestaction.first, Bestaction.second, 0.0;
}

void QLearning::training()
{
    int maingoal = 0;

#pragma omp parallel for
    for (int i = 0; i < 1000; ++i)
    {
        HaxBall env;
        int goal = 0;
        Eigen::VectorXd
            state(env.getStateDimension()),
            action(env.getActionDimension()),
            state_prime(env.getStateDimension());

        for (int j = 0; j < 100; ++j)
        {
            env.getState(state);
            std::pair<int, int> RoundedState = roundedState(state);
            std::pair<int, int> Bestaction = getBestAction(qTable, RoundedState);
            policy(state, action);
            env.step(action);
            env.getState(state_prime);
            std::pair<int, int> RoundedState_prime = roundedState(state_prime);
            updateQTable(state, qTable, RoundedState, Bestaction, RoundedState_prime, 0.1, 0.1);

            // if (state[2] >= 3.95 && state[3] >= -0.7 && state[3] <= 0.7)
            // {
            //     goal++;
            // }
        }

#pragma omp critical
        {
            maingoal += goal;
        }
    }

   // outfile << static_cast<double>(maingoal) / 1000.0 << std::endl;
}


void QLearning::policy(const Eigen::Ref<const Eigen::VectorXd>& state,
                       Eigen::Ref<Eigen::VectorXd> action) const
{
  // Implementation for the linear policy using state and action
  std::pair<int, int> RoundedState = roundedState(state);
  std::pair<int, int> Bestaction = getBestAction(qTable, RoundedState);
  action << Bestaction.first, Bestaction.second, 1.0;
}

void QLearning::policy(const Eigen::Ref<const Eigen::VectorXd>& state,
                       const Eigen::Ref<const Eigen::VectorXd>& parameters,
                       Eigen::Ref<Eigen::VectorXd> action) const
{
  // Implementation for the linear policy using state, parameters, and action
}

void QLearning::Qpolicy(const Eigen::Ref<const Eigen::VectorXd>& state,
                        Eigen::Ref<Eigen::VectorXd> action) const
{
  // Implementation for the Q policy using state and action
}

double QLearning::reward(const Eigen::Ref<const Eigen::VectorXd>& s,
                         const Eigen::Ref<const Eigen::VectorXd>& action,
                         const Eigen::Ref<const Eigen::VectorXd>& s_prime) const
{
  // Implementation for the reward function
  // Return the one-step reward for the given (s, action, s_prime) tuple
  return 42.0f;
}

double QLearning::getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state,
                             const Eigen::Ref<const Eigen::VectorXd>& action) const
{
  // Implementation for the getQfactor function
  // Return the Q-factor for the given (state, action) tuple
  return 42.0f;
}

Eigen::VectorXd QLearning::getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state) const
{
  // Implementation for the getQfactor function
  // Return the Q-factors for all available actions in the given state
  Eigen::Vector4d Q;  // I took 4 components without particular reason

  Q.fill(42.0);

  return Q;
}