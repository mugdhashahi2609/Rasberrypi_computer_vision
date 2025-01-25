#include "task_manager.h"
#include <iostream>
#include <stdexcept>
#include <evl/evl.h>

TaskManager::TaskManager(const std::string& name, int priority, int period_ns)
    : name_(name), priority_(priority) {
    period_.tv_sec = period_ns / 1e9;
    period_.tv_nsec = period_ns % static_cast<int>(1e9);

    if (evl_attach_thread(EVL_CLONE_PRIVATE, name.c_str(), &thread_) < 0) {
        throw std::runtime_error("Failed to attach thread: " + name);
    }
}

void TaskManager::start() {
    std::cout << "Starting task: " << name_ << std::endl;
    evl_set_schedattr(&thread_, priority_);
}

void TaskManager::stop() {
    std::cout << "Stopping task: " << name_ << std::endl;
    evl_detach_thread(&thread_);
}

void TaskManager::executeTask() {
    std::cout << "Executing task: " << name_ << std::endl;
    // Simulate task execution
    evl_sleep(&period_);
}

bool TaskManager::checkDeadlineMiss() {
    return evl_read_tp(&thread_) != 0; // Check thread overruns
}
