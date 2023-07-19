#include "HaxBall.h"

#include <iostream>
#include <chrono>   // For seeding under windows and mingw, if the random device is not bug free

#include <QDebug>

HaxBall::HaxBall(bool has_opponent, QObject* parent) : QObject(parent),
  // m_random_engine(m_random_device()),
  m_random_engine(static_cast<long unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count())),
  m_uniform_dist(0.0, 1.0),
  m_radius_player(0.25), m_radius_ball(0.15), m_distance_goalkeeper(1.0),
  m_dt(0.05), m_sub_dt(0.1 * m_dt),
  m_max_speed_player(2.0), m_max_speed_ball(3.0 * m_max_speed_player), m_friction(0.996),
  m_shoot_dist(2.0 * m_radius_ball),
  m_wasInLeftGoal(false), m_wasInRightGoal(false),
  m_has_opponent(has_opponent), m_num_agent_goals(0), m_num_opponent_goals(0)
{
  // Constructor needs the top left corner and the extent
  m_size = QRectF(-4.0, -2.0, 8.0, 4.0);

  m_goal_left = QRectF(QPointF(0.9 * m_size.left() - 0.15, 0.25 * m_size.top() - 0.0),
                       QPointF(0.9 * m_size.left() + 0.15, 0.25 * m_size.bottom() + 0.0));

  m_goal_right = QRectF(QPointF(0.9 * m_size.right() - 0.15, 0.25 * m_size.top() - 0.0),
                        QPointF(0.9 * m_size.right() + 0.15, 0.25 * m_size.bottom() + 0.0));

  m_position_opponent << 0.0, 0.0;
  m_velocity_opponent << 0.0, 0.0;
  m_goal_right_center << m_goal_right.center().x(), m_goal_right.center().y();

  m_state.resize(getStateDimension());

  // For debugging you could set a specific start state here
  // or set lateron via setState() a desired state from the outside
  //  m_state << m_size.left(), m_size.top(),
  //             1.0, 0.0,
  //             -m_max_speed_ball, 0.0;

  reset();
}
HaxBall::~HaxBall(){}

qreal HaxBall::getTimeDelta() const{ return m_dt; }
qreal HaxBall::getSubTimeDelta() const{ return m_sub_dt; }

qreal HaxBall::getMaxSpeedBall() const{ return m_max_speed_ball; }
qreal HaxBall::getMaxSpeedPlayer() const { return m_max_speed_player; }

QRectF HaxBall::getSize() const {return m_size;}
QRectF HaxBall::getGoalLeft() const {return m_goal_left;}
QRectF HaxBall::getGoalRight() const {return m_goal_right;}

qreal HaxBall::getOpponentDistance() const { return m_distance_goalkeeper; }
qreal HaxBall::getShootingDistance() const { return m_shoot_dist; }

qreal HaxBall::getRadiusBall() const{return m_radius_ball; }
qreal HaxBall::getRadiusPlayer() const{return m_radius_player; }

QPointF HaxBall::getPlayerPos() const{ return QPointF(m_state(0), m_state(1)); }
QPointF HaxBall::getBallPos() const{ return QPointF(m_state(2), m_state(3)); }
QPointF HaxBall::getOpponentPos() const { return QPointF(m_position_opponent(0), m_position_opponent(1)); }

bool HaxBall::ballWasInLeftGoal() const { return m_wasInLeftGoal; }
bool HaxBall::ballWasInRightGoal() const { return m_wasInRightGoal; }

void HaxBall::getState(Eigen::Ref<Eigen::VectorXd> state) const { state = m_state; }
void HaxBall::setState(const Eigen::Ref<const Eigen::VectorXd>& state)
{
  if( state(0) < m_size.left() or state(0) > m_size.right() or
      state(1) < m_size.top() or state(1) > m_size.bottom() or
      state(2) < m_size.left() or state(2) > m_size.right() or
      state(3) < m_size.top() or state(3) > m_size.bottom() or
      state(4) < -m_max_speed_ball or state(4) > +m_max_speed_ball or
      state(5) < -m_max_speed_ball or state(5) > +m_max_speed_ball )
  {
    std::stringstream ss;
    ss << "Invalid state for haxball: " << state.transpose();
    throw std::out_of_range(ss.str());
  }

  // Setting the state from the outside invalidates the remaining internal state of the environment
  resetOthers();

  m_state = state;
}
void HaxBall::setState(double player_x, double player_y, double ball_x, double ball_y, double ball_vx, double ball_vy)
{
  // Maybe wasted effort, but allows calling the other function with the range check
  Eigen::VectorXd state(getStateDimension());
  state << player_x, player_y, ball_x, ball_y, ball_vx, ball_vy;
  setState(state);
}

void HaxBall::step(const Eigen::Ref<const Eigen::VectorXd>& action)
{
  for (double t = 0.0; t < m_dt; t += m_sub_dt)
  {
    subStep(action);
  }
}

void HaxBall::subStep(const Eigen::Ref<const Eigen::VectorXd>& action)
{
  Eigen::Vector2d player_pos = m_state.segment(0, 2);
  Eigen::Vector2d ball_pos = m_state.segment(2, 2);
  Eigen::Vector2d ball_vel = m_state.segment(4, 2);

  // Determines the size of the action space
  const double bound = 1.0;

  // Parse actions: Get player velocity in first 2 components and shooting indicator in last component
  // Also applies clipping to intended action space
  Eigen::Vector2d player_vel = m_max_speed_player * action.segment(0, 2).array().min(bound).max(-bound);
  bool shoot = action(2) > 0.5;

  // friction is a percentage, i.e., what part of the velocity "survives"
  // Currently, there is no integration of accelerations required, e.g. wind.
  ball_vel = ball_vel * m_friction;

  // Euler integration for position
  ball_pos = ball_pos + m_sub_dt * ball_vel;
  player_pos = player_pos + m_sub_dt * player_vel;

  if(m_has_opponent)
  {
    // Define opponent position to be between goal and ball
    // Since it is defined exclusively by the player position, the opponent is NOT part of the state vector
    m_position_opponent = player_pos - m_goal_right_center;
    // One meter away from goal center (well if distance is set to 1m)
    m_position_opponent = m_goal_right_center + m_distance_goalkeeper * m_position_opponent / m_position_opponent.norm();
  }

  // handle player-ball collision, player is inf heavy mass so the ball gets the velocity
  collision(player_pos, player_vel, m_radius_player,
            ball_pos, ball_vel, m_radius_ball, 0.33f);

  if(m_has_opponent)
  {
    // handle opponent-ball collision, opponent is inf heavy mass so the ball gets the velocity
    collision(m_position_opponent, m_velocity_opponent, m_radius_player,
              ball_pos, ball_vel, m_radius_ball, 1.0f);

    // handle player-opponent collision, both are infinte heavy ...
    // opponent stays in place, player is projected to the outside
    collision(m_position_opponent, m_velocity_opponent, m_radius_player,
              player_pos, player_vel, m_radius_player, 1.0f);
  }

  // Collision with infinite heavy borders -> reflection for ball
  if( ball_pos(0) < m_size.left() or ball_pos(0) > m_size.right())
    ball_vel(0) *= -1.0;

  if( ball_pos(1) < m_size.top() or ball_pos(1) > m_size.bottom())
    ball_vel(1) *= -1.0;

  // Clip position to keep player and ball on the screen
  if (player_pos(0) < m_size.left()) player_pos(0) = m_size.left();
  if (player_pos(0) > m_size.right()) player_pos(0) = m_size.right();
  if (player_pos(1) < m_size.top()) player_pos(1) = m_size.top();
  if (player_pos(1) > m_size.bottom()) player_pos(1) = m_size.bottom();

  if (ball_pos(0) < m_size.left()) ball_pos(0) = m_size.left();
  if (ball_pos(0) > m_size.right()) ball_pos(0) = m_size.right();
  if (ball_pos(1) < m_size.top()) ball_pos(1) = m_size.top();
  if (ball_pos(1) > m_size.bottom()) ball_pos(1) = m_size.bottom();

  if (ball_vel(0) < -m_max_speed_ball) ball_vel(0) = -m_max_speed_ball;
  if (ball_vel(0) > +m_max_speed_ball) ball_vel(0) = +m_max_speed_ball;
  if (ball_vel(1) < -m_max_speed_ball) ball_vel(1) = -m_max_speed_ball;
  if (ball_vel(1) > +m_max_speed_ball) ball_vel(1) = +m_max_speed_ball;

  // Handle shooting, if ball is in range its velocity gets overwritten
  Eigen::Vector2d diff = ball_pos - player_pos;
  double distance = diff.norm();

  if (shoot and (distance - m_radius_player - m_radius_ball) < m_shoot_dist and distance > 1e-5)
    ball_vel = m_max_speed_ball * diff / distance;

  // Reset ball if stuck with no velocity behind goal keeper
  if(ball_vel.norm() < 1e-5 and (m_goal_right_center - ball_pos).norm() < m_distance_goalkeeper )
    ball_pos.fill(0.0);

  // Store this indicators to be able to tell what happened during the execution of step()
  m_wasInLeftGoal = m_goal_left.contains(ball_pos(0), ball_pos(1));
  m_wasInRightGoal = m_goal_right.contains(ball_pos(0), ball_pos(1));

  if(m_wasInLeftGoal) m_num_opponent_goals++;
  if(m_wasInRightGoal) m_num_agent_goals++;

  // Reset ball position to center without velocity if it touches a goal area
  if(m_wasInLeftGoal or m_wasInRightGoal)
  {
    ball_pos.fill(0.0);
    ball_vel.fill(0.0);
  }

  // Store the updated state as single vector (for the agent to process)
  // Not using the setter here to avoid the unneccessary boundary check and the call to resetOthers()
  m_state <<
      player_pos(0), player_pos(1),
      ball_pos(0), ball_pos(1),
      ball_vel(0), ball_vel(1);

  return;
}

void HaxBall::reset()
{
  m_state(0) = random_number(m_size.left(), m_size.right());
  m_state(1) = random_number(m_size.top(), m_size.bottom());

  m_state(2) = random_number(m_size.left(), m_size.right());
  m_state(3) = random_number(m_size.top(), m_size.bottom());

  m_state(4) = random_number(-m_max_speed_ball, +m_max_speed_ball);
  m_state(5) = random_number(-m_max_speed_ball, +m_max_speed_ball);

  // Reset again if player is stuck behind goal keeper
  if((m_goal_right_center - m_state.segment(0, 2)).norm() < m_distance_goalkeeper)
    reset();

  resetOthers();
}

bool HaxBall::hasOpponent() const { return m_has_opponent; }

void HaxBall::resetOthers()
{
  m_wasInLeftGoal = false;
  m_wasInRightGoal = false;

  m_num_agent_goals = 0;
  m_num_opponent_goals = 0;
}

int HaxBall::getAgentGoals() const { return m_num_agent_goals; }
int HaxBall::getOpponentGoals() const { return m_num_opponent_goals; }

void HaxBall::collision(Eigen::Ref<Eigen::Vector2d> p1, Eigen::Ref<Eigen::Vector2d> v1, double r1,
                        Eigen::Ref<Eigen::Vector2d> p2, Eigen::Ref<Eigen::Vector2d> v2, double r2,
                        const float elasticity) const
{
  // Work in particel 1's coordinate frame to make things easier
  Eigen::Vector2d p2_in_1 = p2 - p1;
  Eigen::Vector2d v2_in_1 = v2 - v1;

  double distance = p2_in_1.norm();
  
  // Early out such that p2 and v2 remain unchanged, applies to:
  // - No collision
  // - ball and player at same position (avoid zero division)
  if (distance > r1 + r2 or distance < 1e-5)
    return;
  
  // A rotated coordinate system, which makes resolving the collision trivial
  Eigen::Vector2d normal = p2_in_1 / distance;
  Eigen::Vector2d perpen; perpen << -normal(1), normal(0);

  // Project the movement of Particle 2 on this coordinate system
  double v_normal = normal.dot(v2_in_1);
  double v_perpen = perpen.dot(v2_in_1);

  // Reflections == Flipped normal velocity
  // 100% Elastic: Ball gets full velocity
  //   0% Elastic: Ball looses velocity, it goes completely to inf havy particle p1
  Eigen::Vector2d v2_new_normal = -v_normal * normal * elasticity;

  // Perpendicular direction is not affected
  Eigen::Vector2d v2_new_perpen = v_perpen * perpen;

  // Superposition is new velocity after collision
  Eigen::Vector2d v2_new = v2_new_normal + v2_new_perpen;

  // Back to world frame and update the reference
  v2 = v2_new + v1;

  // Finally, fix intersection by teleporting particle p2 to the surface
  p2 = p1 + (r1 + r2) * normal;
}

double HaxBall::random_number(double low, double high)
{
  // Todo: Use the proper distribution parameter info object here
  return m_uniform_dist(m_random_engine) * (high - low) + low;
}

