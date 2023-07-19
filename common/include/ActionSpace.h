#ifndef ACTIONSPACE_H
#define ACTIONSPACE_H

#include "Eigen/Dense"

#include <vector>

namespace Action
{
  ///
  /// \brief action_map Trivial mapping from discrete actions to a continuous ones
  /// \param action discrete action index
  /// \param action_mapped corresponding continuous action as eigen reference
  ///
  /// This function realises a trivial discrete action space.
  /// Of course, there is no need to use all these actions.
  /// Think as part of your initial report about what you really need!
  ///
  void action_map(const int action, Eigen::Ref<Eigen::VectorXd> action_mapped);

  ///
  /// \brief action_map Trivial mapping from continuous actions to a discrete one
  /// \param action continuous action
  /// \return the discrete action
  ///
  /// Thus function realise a trivial discrete action space.
  /// Of course, there is no need to use all these actions.
  /// Think as part of your initial report about what you really need!
  ///
  int action_map(const Eigen::Ref<const Eigen::VectorXd>& action);
}

#endif // ACTIONSPACE_H
