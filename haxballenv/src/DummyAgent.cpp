#include "DummyAgent.h"

#include <cmath>
#include <iostream>

#include <QDebug>

#include <omp.h>

DummyAgent::DummyAgent()
{

}
DummyAgent::~DummyAgent()
{

}

void DummyAgent::policy(const Eigen::Ref<const Eigen::VectorXd>& state,
                        Eigen::Ref<Eigen::VectorXd> action) const
{
  // Your tasks for this method:
  // - conversion of the continuous state vector to your representation of the state space (e.g. grid or features)
  // - querying the value or Q-function for all actions in that state
  // - returning the action with the largest (approximate) Q-value
  //
  // Keep this policy determinstic, i.e., any eps-greedy manipulation of the action goes directly into the training code.
  // - The policy might get used by several threads. Hence the call must be thread-safe -> no shared random number generator
  // - Accessing a persistent random number generator is prevented by the const modifier of the method
  // - You do not want to create and destroy a local generator in every call for the sake of speed

  action << 1.0, 0.0, 1.0;

  return;
}

double DummyAgent::reward(const Eigen::Ref<const Eigen::VectorXd>& state,
                          const Eigen::Ref<const Eigen::VectorXd>& action,
                          const Eigen::Ref<const Eigen::VectorXd>& state_prime) const
{
  // Put your reward signal in here (or call some other function in the directory with shared code)

  // Only rely on information, which is part of s, a and s', and which is still included after your conversion from continuous vectors to something else
  // You can make use of m_world, but only to get constant stuff: the goal position or size of the field.
  // The const-modifer protects you from race conditions

  // Pay attention to not create a non-stationary reward function on accident. In particular for grid based agents:
  //  - the continuous state changes within the same cell, because the grid aggregates information
  //  - if you base the distance calculation on the continuous state, the reward appears changing to the agent's discrete representation of the state
  //  - the fundamental assumption of stationary reward functions is violated
  //  - if this is a problem depends on the actual code / implementation, minor "noise" should be fine

  return 42.0;
}

double DummyAgent::getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state,
                              const Eigen::Ref<const Eigen::VectorXd>& action) const
{
  return 42.0f;
}

Eigen::VectorXd DummyAgent::getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state) const
{
  Eigen::Vector4d Q;

  Q.fill(42.0);

  return Q;
}

void DummyAgent::training(int length, int trajectories, int threads)
{

  // IMPORTANT:
  //  * Only rely on this multithreading approach, if you know what you are doing
  //  * Despite its simplicity this results every your for some groups in a huge chaos
  //  * If you are unsure simply remove the #pragma instruction

  omp_set_num_threads(threads);

#pragma omp parallel for
  for (int i = 0; i < trajectories; ++i)
  {
    training_worker(length);
  }
}

void DummyAgent::training_worker(int length)
{
  // Training code goes here, be aware that multiple threads are active in here
  // The existing code for training is only a proposal, implement whatever you need and do not hesitate to restructure this part
}


