#include "HaxBallGui.h"

#include <QWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QApplication>
#include <QDesktopWidget>
#include <QDir>

#include <cmath>

#include <iostream>
#include <QDebug>

HaxBallGui::HaxBallGui(const BaseAgent& agent, std::shared_ptr<HaxBall> world, QWidget* parent, Qt::WindowFlags flags): QMainWindow(parent, flags),
  m_world(world), m_agent(agent),
  m_timer(0), m_render_step_counter(0), m_render_step_max(1000),
  m_view(0), m_scene(0),
  m_player(0), m_player_indicator(0), m_ball(0), m_opponent(0),
  m_keyboard_policy_active(false),
  m_move_up(false), m_move_down(false), m_move_left(false), m_move_right(false), m_shoot(false),
  m_record_png(false)
{
  QWidget* centeral_widget = new QWidget(this);
  this->setCentralWidget(centeral_widget);

  QVBoxLayout* layout_master = new QVBoxLayout(this);
  centeral_widget->setLayout(layout_master);

  m_view = new QGraphicsView(this);
  layout_master->addWidget(m_view);

  m_scene = new QGraphicsScene(this);
  m_view->setScene(m_scene);

  QHBoxLayout* layout_counter = new QHBoxLayout(this);
  layout_master->addLayout(layout_counter);

  m_goalcounter_agent = new QLCDNumber(this);
  m_goalcounter_opponent = new QLCDNumber(this);

  m_goalcounter_agent->setSegmentStyle(QLCDNumber::SegmentStyle::Flat);
  m_goalcounter_agent->setFrameStyle(QFrame::Shape::NoFrame);

  m_goalcounter_opponent->setSegmentStyle(QLCDNumber::SegmentStyle::Flat);
  m_goalcounter_opponent->setFrameStyle(QFrame::Shape::NoFrame);

  layout_counter->addWidget(m_goalcounter_agent);
  layout_counter->addWidget(m_goalcounter_opponent);

  m_keyboardCheckbox = new QCheckBox("Activate keyboard input", this);

  layout_master->addWidget(m_keyboardCheckbox);
  m_keyboardCheckbox->setChecked(false);
  m_keyboardCheckbox->setFocusPolicy(Qt::FocusPolicy::NoFocus);

  createSceneContent();

  m_timer = new QTimer(this);
  connect(m_timer, &QTimer::timeout, this, &HaxBallGui::playGameStep);
  connect(m_keyboardCheckbox, &QCheckBox::clicked, this, &HaxBallGui::setKeyBoardPolicyActive);

  // Bigger windows, thus bigger game, for high dpi screens. Typically they are wide, hence use the with as indicator
  if (QApplication::desktop()->screenGeometry().width() > 2000)
    this->resize(800, 500);
  else
    this->resize(400, 300);

  this->setWindowTitle("Hax Ball");
}

HaxBallGui::HaxBallGui(const BaseAgent& agent, QWidget* parent, Qt::WindowFlags flags):
  HaxBallGui(agent, std::make_shared<HaxBall>(), parent, flags) {}

HaxBallGui::~HaxBallGui()
{

}

void HaxBallGui::playGame(double speedup, int steps, bool start_with_keyboard, bool record_png)
{
  if(start_with_keyboard and not m_keyboardCheckbox->isChecked())
     m_keyboardCheckbox->click();

  m_record_png = record_png;

  m_render_step_max = steps > 0 ? steps : 1'000'000;

  m_render_step_counter = 0;

  m_timer->start(1000.0 /* ms */ * m_world->getTimeDelta() / speedup );
}

void HaxBallGui::keyPressEvent(QKeyEvent* event) { handleKeyEvent(event, true); }
void HaxBallGui::keyReleaseEvent(QKeyEvent* event){ handleKeyEvent(event, false); }

void HaxBallGui::createSceneContent()
{
  m_scene->clear();

  // Pens and brushes as needed below
  QBrush b_blue(Qt::blue);
  QBrush b_red(Qt::red);
  QBrush b_green(Qt::green);
  QBrush b_black(Qt::black);
  QBrush b_white(Qt::white);
  QBrush b_trans(Qt::transparent);
  QBrush b_grey(Qt::gray);

  QPen p_blue(b_blue, 0.05);
  QPen p_blue_thin(b_blue, 0.0125);
  QPen p_red(b_red, 0.05);
  QPen p_grey(b_grey, 0.05);
  QPen p_black(b_black, 0.05);
  QPen p_black_thin(b_black, 0.0125);
  QPen p_white(b_white, 0.05);

  QRectF s = m_world->getSize();
  qreal r, r2;

  // Green background
  m_scene->addRect(s, p_black, b_green);

  // Goal area
  m_scene->addRect(m_world->getGoalLeft(), p_white, b_white );
  m_scene->addRect(m_world->getGoalRight(), p_white, b_white );

  // The two white rectangles (appears as one big rectangle with vertical line in the middle)
  m_scene->addRect(QRectF(QPointF(0.9 * s.left(), 0.9 * s.top()),
                          QPointF(0.0, 0.9 * s.bottom())),
                   p_white, b_trans);
  m_scene->addRect(QRectF(QPointF(0.0, 0.9 * s.top()),
                          QPointF(0.9 * s.right(), 0.9 * s.bottom())),
                   p_white, b_trans);

  // The circle in the middle
  m_scene->addEllipse(-1.0, -1.0, 2.0, 2.0, p_white, b_trans);

  // The circle on which the goalkeeper is moving
  QPointF c = m_world->getGoalRight().center();
  r = m_world->getOpponentDistance();
  m_scene->addEllipse(c.x()-r, c.y()-r, 2.0*r, 2.0*r, p_black_thin);

  // Player and ball, result is stored in class to be able to update the position
  // Spawn centered at zero position, otherwise you have to compensate the offset when setting the position later on
  r = m_world->getRadiusPlayer();
  r2 = m_world->getShootingDistance() + r;
  m_player = m_scene->addEllipse(-r, -r, 2.0 * r, 2.0 * r, p_blue, b_blue);
  m_player_indicator = m_scene->addEllipse(-r2, -r2, 2.0*r2, 2.0*r2, p_blue_thin );

  r = m_world->getRadiusBall();
  m_ball = m_scene->addEllipse(-r, -r, 2.0 * r, 2.0 * r, p_grey, b_grey);

  if( m_world->hasOpponent())
  {
    r = m_world->getRadiusPlayer();
    m_opponent = m_scene->addEllipse(-r, -r, 2.0 * r, 2.0 * r, p_red, b_red);
  }
}

void HaxBallGui::showEvent(QShowEvent*){fitInView();}
void HaxBallGui::resizeEvent(QResizeEvent* event){fitInView();}

void HaxBallGui::fitInView()
{
  m_scene->setSceneRect(m_world->getSize());
  m_view->fitInView(m_world->getSize());
}

void HaxBallGui::updateViewFromWorld()
{
  if(m_player) m_player->setPos(m_world->getPlayerPos());
  if(m_player_indicator) m_player_indicator->setPos(m_world->getPlayerPos());
  if(m_ball) m_ball->setPos(m_world->getBallPos());
  if(m_opponent) m_opponent->setPos(m_world->getOpponentPos());
}

void HaxBallGui::keyboardPolicy(Eigen::Ref<Eigen::VectorXd> action) const
{
  double dx, dy, dz;

  // Binary indicator to double
  dz = (m_shoot) ? 1.0 : 0.0;

  // Left or right pressed, otherwise zero movement
  if(m_move_left and !m_move_right)
    dx = -1.0;
  else if (!m_move_left and m_move_right)
    dx = 1.0;
  else
    dx = 0.0;

  // Up or down pressed, otherwise zero movement
  // Pay attention to coordinate system
  if(m_move_up and !m_move_down)
    dy = -1.0;
  else if (!m_move_up and m_move_down)
    dy = 1.0;
  else
    dy = 0.0;

  // Reference cannot be resized, it is up to the caller
  action << dx, dy, dz;
}

void HaxBallGui::handleKeyEvent(QKeyEvent* event, bool pressed)
{
  if (event->key() == Qt::Key::Key_W)
     m_move_up = pressed;

  if (event->key() == Qt::Key::Key_S)
    m_move_down = pressed;

  if (event->key() == Qt::Key::Key_A)
    m_move_left = pressed;

  if (event->key() == Qt::Key::Key_D)
    m_move_right = pressed;

  if (event->key() == Qt::Key::Key_R and not event->isAutoRepeat() and not pressed)
    m_world->reset();

  if (event->key() == Qt::Key::Key_Q and not event->isAutoRepeat() and not pressed)
    this->close();

  if (event->key() == Qt::Key::Key_Space || event->key() == Qt::Key::Key_Q || event->key() == Qt::Key::Key_E)
    m_shoot = pressed;
}

void HaxBallGui::playGameStep()
{ 
  // Render the current state in which the action is executed, not the state after the execution
  updateViewFromWorld();

  Eigen::VectorXd
      state(m_world->getStateDimension()),
      action(m_world->getActionDimension()),
      state_prime(m_world->getStateDimension());

  m_world->getState(state);
  // std::cout << state.transpose() << std::endl;

  if (m_keyboard_policy_active)
    keyboardPolicy(action);
  else
    m_agent.policy(state, action);

  m_world->step(action);

  m_world->getState(state_prime);

  double r = m_agent.reward(state, action, state_prime);

  // qDebug() << "r" << r << "Q(s,a)" << m_agent.getQfactor(state, action);
  // std::cout << "Q(s,.) " << m_agent.getQfactor(state) << "\n" << std::endl;

  m_goalcounter_agent->display(m_world->getAgentGoals());
  m_goalcounter_opponent->display(m_world->getOpponentGoals());

  // The subsampling is required to reduce the number png -> smaller gif
  // == 1 to skip the first frame, where the opponent is still at the wrong location
  if(m_record_png and m_render_step_counter % 25 == 1)
  {
    const QString image_dir("images");

    QDir cur(".");
    cur.mkdir(image_dir);

    const int required_digits = std::log10(m_render_step_max) + 1;

    const QString filename = QString("%1/scene_%2.png").arg(image_dir).arg(m_render_step_counter, required_digits, 10, QLatin1Char('0'));

    // Transparent pnq with size of current scene
    QImage png(width(), height(), QImage::Format_ARGB32);
    png.fill(Qt::transparent);

    QPainter painter(&png);
    m_scene->render(&painter);

    png.save(filename);
  }

  m_render_step_counter++;

  if(m_render_step_counter >= m_render_step_max)
  {
    // Don't know if this good, let me know what you think!
    // My guess is that one wants to record a game exactly once and prevent undesired overwriting with the next trajectory / game instance
    if (m_record_png)
    {
      qDebug() << " Terminating to prevent overriding pngs";
      QApplication::quit();
      std::exit(0);
    }
    else
      this->close();

    // These steps should no be required, since this instance of the gui is destroyed anyway
    m_render_step_counter = 0;
    m_record_png = false;
    m_timer->stop();
  }
}

void HaxBallGui::setKeyBoardPolicyActive(bool is_active) { m_keyboard_policy_active = is_active; }
