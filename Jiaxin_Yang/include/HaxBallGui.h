#ifndef _HAXBALLGUI_H_
#define _HAXBALLGUI_H_

#include <memory>

#include <QMainWindow>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsItem>
#include <QLCDNumber>
#include <QTimer>
#include <QKeyEvent>
#include <QCheckBox>

#include "HaxBall.h"
#include "BaseAgent.h"

///
/// \brief The HaxBallGui to visualize a HaxBall environemnt
///
/// This class is responsible for setting up the Qt part and rendering the game and your policiy.
/// It uses a dedicated HaxBall instance to not mess up your agent.
///
/// The rendering of the HaxBall environment is inspired and partially based on a student project from ARL 2020.
/// For details ask Martin Gottwald.
///
class HaxBallGui : public QMainWindow
{
  Q_OBJECT

public:

  ///
  /// \brief Creates a new window for rendering
  /// \param agent the constant reference to your agent, i.e., the policy you want to execute
  /// \param world the shared pointer to the world in which the agent operates
  /// \param parent the parent of this MainWindow (if any)
  /// \param flags flags for this QMainWindow (if any)
  ///
  /// This constructor takes an existing agent and world and runs them with some rendering.
  /// The world you provide mostly varies between whether or not there is an opponent.
  ///
  explicit HaxBallGui(const BaseAgent& agent, std::shared_ptr<HaxBall> world, QWidget* parent=0, Qt::WindowFlags flags = Qt::WindowFlags());
  ///
  /// \brief Creates a new window for rendering
  /// \param agent the constant reference to your agent, i.e. the policy you want to execute
  /// \param parent the parent of this MainWindow (if any)
  /// \param flags flags for this QMainWindow (if any)
  ///
  /// \overload
  ///
  explicit HaxBallGui(const BaseAgent& agent, QWidget* parent=0, Qt::WindowFlags flags = Qt::WindowFlags());

  virtual ~HaxBallGui();

  ///
  /// \brief playGame starts a new game
  /// \param speedup a factor to speed up the execution by reducing the waiting time between frames
  /// \param steps how many steps to play
  ///
  void playGame(double speedup = 1.0, int steps = 5000, bool start_with_keyboard = false, bool record_png = false);

protected:

  void keyPressEvent(QKeyEvent* event);

  void keyReleaseEvent(QKeyEvent* event);

private:

  ///
  /// \brief createSceneContent Fills the internal scene
  ///
  /// Call this function to create the graphics items.
  /// Their initial position may not match the current state of the HaxBall environment
  ///
  void createSceneContent();

  ///
  /// \brief showEvent Overriden to update the view on showing the window
  /// \param event contains details for the event
  ///
  void showEvent(QShowEvent* event);

  ///
  /// \brief resizeEvent Overriden to update the view on resizing the window
  /// \param event contains details for the event
  ///
  void resizeEvent(QResizeEvent* event);

  ///
  /// \brief fitInView Updates the graphics scene to the current window size
  ///
  /// Call this function to strecht the HaxBall game to the current window size.
  ///
  void fitInView();

  ///
  /// \brief updateViewFromWorld Updates the moving parts based on the HaxBall instance
  ///
  /// The position of the moving parts is set to the position as in the HaxBall environment
  /// This method checks, wether the pointers to the graphic elements are non zero
  ///
  void updateViewFromWorld();

  ///
  /// \brief Handles a key event
  /// \param event the event
  /// \param pressed whether or not the butten was pressed
  ///
  /// This function checks the event for all relevant keys and updates the internal state
  ///
  void handleKeyEvent(QKeyEvent* event, bool pressed);

  /// \brief Policy based on the keyboard
  /// \param action the action according to the keystrokes
  ///
  /// Hardcoded actions, which are produced by pressing the corresponding key
  ///
  void keyboardPolicy(Eigen::Ref<Eigen::VectorXd> action) const;

private slots:

  ///
  /// \brief playGameStep Gathers an action from the policy, executes it and updates the view
  ///
  /// This method is the main loop to play HaxBall and gets called by the internal timer:
  /// - Update the visualization
  /// - Querry an action by sending the current state to the policy method of the agent
  /// - Execute the action
  /// - Manage the timer and start over until engough steps were done
  ///
  void playGameStep();

  void setKeyBoardPolicyActive(bool is_active);

private:

  /// This HaxBall instance is used for testing an agent and to querry the rendering details
  std::shared_ptr<HaxBall> m_world;

  /// This constant reference to an agent is the agent to test. It is constant to prevent change to the internal state during testing (e.g. manipulating the random engine)
  const BaseAgent& m_agent;

  // Parts of the GUI, nothing important below

  QTimer* m_timer;
  int m_render_step_counter, m_render_step_max;

  QLCDNumber *m_goalcounter_agent, *m_goalcounter_opponent;

  QGraphicsView* m_view;
  QGraphicsScene* m_scene;

  QCheckBox* m_keyboardCheckbox;

  QGraphicsItem *m_player, *m_player_indicator, *m_ball, *m_opponent;

  // If true, the gui uses the keyboard policy
  bool m_keyboard_policy_active;

  // The state of the buttons for creating actions by hand
  bool m_move_up, m_move_down, m_move_left, m_move_right, m_shoot;

  // If set to true, each step produces one png
  bool m_record_png;
};

#endif // _HAXBALLGUI_H_
