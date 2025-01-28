#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "task_manager.h"
#include <vector>

class Scheduler {
public:
    void addTask(TaskManager& task);
    void run();
    void restartIndividualTask(TaskManager& task);
    void restartAllTasks();

private:
    std::vector<TaskManager> tasks_;
};

#endif // SCHEDULER_H
