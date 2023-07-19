#ifndef _HAXBALL_H_
#define _HAXBALL_H_

#include <random>

#include <QObject>
#include <QRectF>

#include "Eigen/Dense"

///
/// \brief The HaxBall class with the differential equations
///
/// This class contains the game HaxBall and the simulation code.
/// You can create as many instances as you land use them for training.
///
/// This class uses QT code, because QRectF and others are quite handy.
/// Also the signal slot mechanism is available, let me know what you need and I add it here.
///
/// Important: The game uses Qt's coordinate system
///   +------> x
///   |
///   |
///   V
///   y
///
/// The environment is inspired and partially based on a student project from ARL 2020.
/// For details ask Martin Gottwald.
///
class HaxBall: public QObject
{
  Q_OBJECT

public:
  ///
  /// \brief HaxBall Creates a new HaxBall environment
  /// \param has_opponent Enables or disables the opponent
  /// \param parent the parent of this QObject (if any)
  ///
  /// Contains only the initialization of all values and creates the starting state by resetting the environment
  ///
  HaxBall(bool has_opponent = true, QObject* parent = 0);
  virtual ~HaxBall();

  ///
  /// \brief getTimeDelta
  /// \return the elapsing time between two steps
  ///
  qreal getTimeDelta() const;

  ///
  /// \brief getSubTimeDelta
  /// \return the elapsing time between two sub steps
  ///
  qreal getSubTimeDelta() const;

  ///
  /// \brief getMaxSpeedBall
  /// \return the maximum speed of the ball
  ///
  /// The maximum speed is the same for x and y axis.
  /// So the actual max speed is bigger along the diagonals ...
  ///
  qreal getMaxSpeedBall() const;

  ///
  /// \brief getMaxSpeedPlayer
  /// \return the maximum speed of the player
  ///
  /// The maximum speed is the same for x and y axis.
  /// So the actual max speed is bigger along the diagonals ...
  ///
  qreal getMaxSpeedPlayer() const;

  ///
  /// \brief getSize
  /// \return the size of the playing ground
  ///
  QRectF getSize() const;

  ///
  /// \brief getGoalLeft
  /// \return the rectangle corresponding to the left (player) goal
  ///
  QRectF getGoalLeft() const;

  ///
  /// \brief getGoalLeft
  /// \return the rectangle corresponding to the right (opponent) goal
  ///
  QRectF getGoalRight() const;

  ///
  /// \brief getOpponentDistance
  /// \return the distance between the goalkeeper and its goal center
  ///
  /// Used for rendering.
  ///
  qreal getOpponentDistance() const;

  ///
  /// \brief getShootingDistance
  /// \return the distance between the player and the ball in which shooting is possible
  ///
  /// Used for rendering.
  ///
  qreal getShootingDistance() const;

  ///
  /// \brief getRadiusBall
  /// \return the radius of the ball (not the diameter)
  ///
  qreal getRadiusBall() const;

  ///
  /// \brief getRadiusPlayer
  /// \return the radius of the player (not the diameter)
  ///
  qreal getRadiusPlayer() const;

  ///
  /// \brief getPlayerPos
  /// \return the position of the player as QPointF
  ///
  /// For training it should be better to get once the Eigen::VectorXd and process the state in the agent.
  /// This getter is for plotting.
  ///
  QPointF getPlayerPos() const;

  ///
  /// \brief getBallPos
  /// \return the position of the ball as QPointF
  ///
  /// For training it should be better to get once the Eigen::VectorXd and process the state in the agent.
  /// This getter is for plotting.
  ///
  QPointF getBallPos() const;

  ///
  /// \brief getOpponentPos
  /// \return the position of the opponent as QPointF
  ///
  /// This getter is for plotting.
  ///
  QPointF getOpponentPos() const;

  ///
  /// \brief ballInLeftGoal
  /// \return true, if the ball hit the left goal area during the execution of a step. Otherwise false.
  ///
  bool ballWasInLeftGoal() const;

  ///
  /// \brief ballInRightGoal
  /// \return true, if the ball hit the right goal area during the execution of a step. Otherwise false.
  ///
  bool ballWasInRightGoal() const;

  ///
  /// \brief getState
  /// \param state An eigen reference to a column vector, it receives a copy of the state vector
  ///
  /// Important:
  /// - References cannot be resized
  /// - it is up to you to provide a vector with correct size (currently getNO() x 1)
  ///
  void getState(Eigen::Ref<Eigen::VectorXd> state) const;
  
  ///
  /// \brief setState
  /// \param state A constant eigen reference to a column vector describing the new environment state
  ///
  /// Some checks are applied.
  ///
  void setState(const Eigen::Ref<const Eigen::VectorXd>& state);

  ///
  /// \brief setState
  /// \param player_x Player position on x-axis
  /// \param player_y Player position on y-axis
  /// \param ball_x Ball position on x-axis
  /// \param ball_y Ball position on y-axis
  /// \param ball_vx Ball velocity on x-axis
  /// \param ball_vy Ball velocity on y-axis
  ///
  /// @overload
  ///
  void setState(double player_x, double player_y, double ball_x, double ball_y, double ball_vx, double ball_vy);

  ///
  /// \brief step Executes the action in the world
  /// \param action the action to execute
  ///
  /// This function should be fail safe, once it returns the internal state is updated.
  /// The step is not thread safe!
  ///
  void step(const Eigen::Ref<const Eigen::VectorXd>& action);

  ///
  /// \brief Returns the dimension of the state space
  /// \return the (hardcoded) number of state dimensions of the control problem
  ///
  int getStateDimension() const { return 6; }

  ///
  /// \brief Returns the dimension of the action space
  /// \return the (hardcoded) number of action dimensions of the control problem
  ///
  int getActionDimension() const { return 3; }

  ///
  /// \brief reset Resets the state to some random value in the state space
  ///
  /// The Player is positioned somewhere on the play ground.
  /// The Ball as well (maybe inside the player or behind the goal keeper ...)
  /// The Ball velocity is somewhere between the max values
  ///
  /// All distributions are uniform.
  ///
  void reset();

  ///
  /// \brief hasOpponent
  /// \return true, if an opponent is present
  ///
  /// Used for plotting to toggle the rendering of an opponent
  ///
  bool hasOpponent() const;

  /// \brief Returns the number of goals of your agent since the last reset
  /// \return The number of agent goals since last reset
  ///
  /// Counts the number of goal since the last reset, do not use this value in your
  /// reward function as the counter depends on states from the past.
  ///
  int getAgentGoals() const;

  /// \brief Returns the number of goals of the opponent since the last reset
  /// \return The number of opponent goals since last reset
  ///
  /// Counts the number of goal since the last reset, do not use this value in your
  /// reward function as the counter depends on states from the past.
  ///
  int getOpponentGoals() const;

private:

  ///
  /// \brief step Executes the action in the world
  /// \param action the action to execute
  ///
  /// This function should be fail safe, once it returns the internal state is updated.
  /// The step is not thread safe!
  ///
  /// The substep executes a fraction of the intended time to elapse.
  /// This mechanism is required for proper collision handling.
  ///
  void subStep(const Eigen::Ref<const Eigen::VectorXd>& action);

  ///
  /// \brief resets other variables outside of the state vector
  ///
  /// This function is responsible to reset additional variables of the environment
  /// such as the was-in-goal indicators.
  ///
  void resetOthers();

  ///
  /// \brief collision Handles the collision between two particles
  /// \param p1 Position of Particle 1
  /// \param v1 Velocity of Particle 1
  /// \param r1 Radius of Particle 1
  /// \param p2 Position of Particle 2
  /// \param v2 Velocity of Particle 2
  /// \param r2 Radius of Particle 2
  ///
  /// Particle 1 is infinte heavy.
  /// Particle 2 has no mass, in bounces perfectly at the surface of particle 1.
  ///
  /// The method corrects penetrations between particles by projecting Particle 2 back to the surface
  /// The vectors remain unchanged if there is no collision (distance > sum of radii)
  ///
  void collision(Eigen::Ref<Eigen::Vector2d> p1, Eigen::Ref<Eigen::Vector2d> v1, double r1,
                 Eigen::Ref<Eigen::Vector2d> p2, Eigen::Ref<Eigen::Vector2d> v2, double r2, const float elasticity) const;

  ///
  /// \brief random_number Produces a random number in given interval
  /// \param low lower limit of interval
  /// \param high higher limit of interval
  /// \return the random number
  ///
  /// Attention: uses a dirty hack instead of the proper std::c++ random machanism with that distribution parameter info thing.
  ///
  double random_number(double low, double high);

private:

  // Stuff for the random number generator, these things are not thread safe -> do not use one haxball environment in multiple threads
  std::random_device m_random_device;
  std::mt19937 m_random_engine;
  std::uniform_real_distribution<> m_uniform_dist;

  // Variables to describe the simulation
  QRectF m_size, m_goal_left, m_goal_right;
  const qreal m_radius_player, m_radius_ball, m_distance_goalkeeper;
  const qreal m_dt, m_sub_dt, m_max_speed_player, m_max_speed_ball, m_friction, m_shoot_dist;

  // Variables to make the work with an agent easier
  bool m_wasInLeftGoal, m_wasInRightGoal;

  // A true column vector with getNO() components
  Eigen::VectorXd m_state;

  // Change its value in the constructor to switch between easy mode or with oppnent
  const bool m_has_opponent;

  // The position of the opponent, it is defined directly from the state vector and can stay outside of m_state
  // The variable must be a member for rendering.
  Eigen::Vector2d m_position_opponent;

  // The velocity of the opponent, currently it is always zero and just required to call the collision function
  Eigen::Vector2d m_velocity_opponent;

  // The right goal center as Eigen::Vector for easy computations
  Eigen::Vector2d m_goal_right_center;

  // Number of goals
  int m_num_agent_goals, m_num_opponent_goals;
};

#endif // _HAXBALL_H_
