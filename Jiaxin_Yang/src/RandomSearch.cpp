#include "RandomSearch.h"

#include <cmath>
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort
#include <QDebug>

#include <omp.h>

#include "RewardFunctions.h"
#include "eigenmvn.h" // Multivariate Normal Distribution

const unsigned int RandomSearch::N_TOTAL = 10'000;
const unsigned int RandomSearch::N_KEEP = 1000;
const unsigned int RandomSearch::TAU = 100;
const double RandomSearch::GAMMA = 0.9;


template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v)
{
  // Argsort https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
  
  // initialize original index locations
  std::vector<size_t> idx(v.size()); 
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

RandomSearch::RandomSearch()
{
  // No clue where to start, but should not matter due to sampling with huge covariance in beginning
  m_parameters.resize(m_world.getStateDimension() * m_world.getActionDimension());
  m_parameters.fill(0.0);
  

  // Sufficiently large to have particles in entire required range
  m_covariance = 1.0 * Eigen::MatrixXd::Identity(m_parameters.rows(), m_parameters.rows());
}
RandomSearch::~RandomSearch()
{

}


void RandomSearch::policy(const Eigen::Ref<const Eigen::VectorXd>& state,
                          Eigen::Ref<Eigen::VectorXd> action) const
{
  policy(state, m_parameters, action);

  return;
}

void RandomSearch::policy(const Eigen::Ref<const Eigen::VectorXd>& state,
                          const Eigen::Ref<const Eigen::VectorXd>& parameters,
                          Eigen::Ref<Eigen::VectorXd> action) const
{
  const unsigned int dim = m_world.getStateDimension();

  double a1,a2,a3;

  // The three chunks of the parameter vector times the state
  // This forms a linear policy pi(s) = a = A * s
  a1 = parameters.segment(0, dim).dot(state);
  a2 = parameters.segment(dim, dim).dot(state);
  a3 = parameters.segment(2*dim, dim).dot(state);

  // Clipping is required to keep the actions in the space, the inner product results in arbitrary large values
  if( a1 >  1.0 ) a1 = 1.0;
  if( a1 < -1.0 ) a1 = -1.0;
  if( a2 >  1.0 ) a2 = 1.0;
  if( a2 < -1.0 ) a2 = -1.0;
  if( a3 >  1.0 ) a3 = 1.0;
  if( a3 <  0.0 ) a3 = 0.0;

  action << a1, a2, a3;
}

double RandomSearch::reward(const Eigen::Ref<const Eigen::VectorXd>& state,
                          const Eigen::Ref<const Eigen::VectorXd>& action,
                          const Eigen::Ref<const Eigen::VectorXd>& state_prime) const
{
  // Switch to the player ball distance reward and see what comes out ...
  return Reward::distance_player_origin(state, action, state_prime);
}

double RandomSearch::getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state,
                              const Eigen::Ref<const Eigen::VectorXd>& action) const
{
  return 42.0f;
}

Eigen::VectorXd RandomSearch::getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state) const
{
  Eigen::Vector4d Q;  // I took 4 components without particular reason

  Q.fill(42.0);

  return Q;
}

void RandomSearch::training()
{
  // Draw particles from multivariate Gaussian:
  // https://ros-developer.com/2017/11/15/generating-multivariate-normal-distribution-samples-using-c11-eigen-library/
  // https://github.com/beniz/eigenmvn
  Eigen::EigenMultivariateNormal<double> normal(m_parameters, m_covariance);
  Eigen::MatrixXd particles = normal.samples(RandomSearch::N_TOTAL);  // 18 x 500

  std::vector<double> scores(RandomSearch::N_TOTAL);

  // Rollouts as in the eval center, but since the policy is changed for each particle there is no easy way to reuse existing code ...
#pragma omp parallel for
  for (int i = 0; i < RandomSearch::N_TOTAL; ++i)
  {
    // One step reward and accumulator for discounted return
    double r, R = 0.0;

    // Required in worker thread to keep the internal states of the environment separated
    // Initialises the environment randomly
    HaxBall env;

    // Variables to store the s,a,s' tuple, one copy per worker thread
    Eigen::VectorXd
        state(env.getStateDimension()),
        action(env.getActionDimension()),
        state_prime(env.getStateDimension());

    // Create rollout (finite horizon approximation for infinite horizon, choose TAU long or GAMMA small enough
    for(int j = 0; j < RandomSearch::TAU ; ++j)
    {
      env.getState(state);
      // std::cout << "State: " << state << std::endl;
      policy(state, particles.col(i), action);
      std::cout << "Action: " << action << std::endl;
      env.step(action);
      env.getState(state_prime);
      // std::cout << "State_prime: " << state_prime << std::endl;
      r = reward(state, action, state_prime);
      R += std::pow(RandomSearch::GAMMA, j) * r;
    }

    scores[i] = R;
  }

  // Sort particles according to their scores
  // idx is result similar to numpy.argsort, sorted from small to large
  std::vector<size_t> idx = sort_indexes<double>(scores);

  // Select elite particles, best ones are located at the end
  Eigen::MatrixXd elite(particles.rows(), RandomSearch::N_KEEP);

  for (int i=0; i<RandomSearch::N_KEEP; ++i)
    elite.col(i) = particles.col(idx[RandomSearch::N_TOTAL - 1 - i]);

  // They form the new mean and covariance
  // https://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
  m_parameters = elite.rowwise().mean();

  Eigen::MatrixXd centered = elite.colwise() - elite.rowwise().mean();
  m_covariance = (centered * centered.adjoint()) / double(RandomSearch::N_KEEP - 1);



  /////// select the best action ////////
}

