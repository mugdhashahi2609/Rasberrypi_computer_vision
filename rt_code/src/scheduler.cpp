#include "scheduler.h"
#include <iostream>

void Scheduler::addTask(TaskManager& task) {
    tasks_.push_back(task);
}

void Scheduler::run() {
    for (auto& task : tasks_) {
        task.start();
    }

    while (true) {
        for (auto& task : tasks_) {
            task.executeTask();
            if (task.checkDeadlineMiss()) {
                std::cerr << "Task " << task.getName() << " missed its deadline!" << std::endl;
                restartIndividualTask(task);
            }
        }
    }
}

void Scheduler::restartIndividualTask(TaskManager& task) {
    std::cerr << "Restarting task: " << task.getName() << std::endl;
    task.stop();
    task.start();
}

void Scheduler::restartAllTasks() {
    std::cerr << "Restarting all tasks!" << std::endl;
    for (auto& task : tasks_) {
        task.stop();
    }
    for (auto& task : tasks_) {
        task.start();
    }
}
