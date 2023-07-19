#include "QLearning.h"
#include <iostream>
#include <QApplication>
#include <QDebug>
#include <Eigen/Dense>
#include "HaxBall.h"
#include "HaxBallGui.h"
#include "EvaluationCenter.h"
#include "RandomSearch.h"
#include "DummyAgent.h"


void playing(const BaseAgent& agent, int argc, char** argv)
{
  QApplication app(argc, argv);
  HaxBallGui gui(agent);
  gui.show();
  gui.playGame(1.0, -1, true, false);
  app.exec();
}

void render(const BaseAgent& agent, int argc, char** argv)
{
  QApplication app(argc, argv);
  HaxBallGui gui(agent);
  gui.show();
  gui.playGame(1.0, 1e3, false, false);
  app.exec();
}

int main(int argc, char** argv)
{
  std::cout << "Hello Group Group-2!" << std::endl;
  QLearning agent;
  // Use the dummy agent to compile the code (= minimal working example)
  // DummyAgent agent;
  // playing(agent, argc, argv);

  // Use the random search agent (based on Cross Entropy Method) as a more sophisticated example
  // RandomSearch agent;
  // Create your own agent and provide the parameters you need, something like:
  // AwesomeAgent agent(/*alpha*/   0.001,
  //                    /*gamma*/   0.99,
  //                    /*epsilon*/ 0.01);

  // The evaluation center provides you with some metrics for the progress
  // Results in a .csv files next to the executable
  EvaluationCenter eval(agent, RandomSearch::GAMMA);

  // This could be a learning loop, extend it as required and make sure, that your
  // agent stores everything on the disk
  // The code below is only a proposal and demonstration, do whatever you need!
  for (int i = 0; i < 500; ++i)
  {
    std::cout << i << std::endl;

    agent.training();
    eval.evaluate();
    if( i % 3 == 1)
      render(agent, argc, argv);
  }

  return 0;
}
