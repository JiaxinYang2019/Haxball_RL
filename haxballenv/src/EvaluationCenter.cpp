#include "EvaluationCenter.h"

#include <iostream>
#include <fstream>
#include <cmath>

EvaluationCenter::EvaluationCenter(const BaseAgent& agent, std::shared_ptr<HaxBall> world, double gamma) :
  m_world(world), m_agent(agent), m_gamma(gamma)
{
  writeHeader();

  createProbes();
}

EvaluationCenter::EvaluationCenter(const BaseAgent& agent, double gamma):
  EvaluationCenter(agent, std::make_shared<HaxBall>(), gamma)
{

}

EvaluationCenter::~EvaluationCenter()
{

}

void EvaluationCenter::evaluate()
{
  Eigen::VectorXd
      state(m_world->getStateDimension()),
      action(m_world->getActionDimension());

  double V, R;

  std::ofstream file;
  file.open("eval.csv", std::ios::out | std::ios::app);

  // Expected reward for all probes according to agent
  for (int i = 0; i < EvaluationCenter::N; ++i)
  {
    state = m_probes[i];
    m_agent.policy(state, action);

    V = m_agent.getQfactor(state, action); // This is not Q but V, since the action is selected according to the policy

    file << V << ",";
  }

  // True Discounted Returns according to rollouts
  for (int i = 0; i < EvaluationCenter::N; ++i)
  {
    R = rollout(m_probes[i]);

    file << R;

    // The last , must be omitted for .csv format
    if(i < EvaluationCenter::N-1)
       file << ",";
  }

  file << std::endl;

  file.close();
}

double EvaluationCenter::rollout(const Eigen::Ref<const Eigen::VectorXd>& start_state)
{
  Eigen::VectorXd
      state(m_world->getStateDimension()),
      action(m_world->getActionDimension()),
      state_prime(m_world->getStateDimension());

  // Prepare environment
  m_world->reset();
  m_world->setState(start_state);

  // Accumulator for the discounted return
  double R = 0.0, r;

  // Create rollout
  for(int j = 0; j < EvaluationCenter::TAU; ++j)
  {
    m_world->getState(state);

    m_agent.policy(state, action);

    m_world->step(action);

    m_world->getState(state_prime);

    r = m_agent.reward(state, action, state_prime);

    R += std::pow(m_gamma, j) * r;
  }

  return R;
}

void EvaluationCenter::writeHeader() const
{

  std::ofstream file;

  file.open("eval.csv", std::ios::out);

  // Estimatation for expected reward, i.e., V(s) = Q(s, pi(s))
  for (int i = 0; i < EvaluationCenter::N; ++i)
  {
    file << "V_" << i << ",";
  }

  // True Discounted Returns
  for (int i = 0; i < EvaluationCenter::N; ++i)
  {
    file << "R_" << i;

    // The last , must be omitted for .csv format
    if(i < EvaluationCenter::N-1)
       file << ",";
  }

  file << std::endl;

  file.close();
}

void EvaluationCenter::createProbes()
{
  m_probes.resize(EvaluationCenter::N);

  for (int i = 0; i < m_probes.size(); ++i)
  {
    m_probes[i].resize(m_world->getStateDimension());

    // Quick and dirty way to get a random state from the game
    m_world->reset();
    m_world->getState(m_probes[i]);
  }

  std::ofstream file;
  file.open("probes.csv", std::ios::out);

  file << "p_x,p_y,b_x,b_y,v_x,v_y" << std::endl;

  for (int i = 0; i < m_probes.size(); ++i)
  {
    for (int j = 0; j < m_world->getStateDimension(); ++j)
    {
      file << m_probes[i](j);

      // The last , must be omitted for .csv format
      if(j < m_world->getStateDimension()-1)
         file << ",";
    }

    file << std::endl;

  }

  file.close();
}
