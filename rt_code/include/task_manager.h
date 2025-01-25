#ifndef TASK_MANAGER_H
#define TASK_MANAGER_H

#include <evl/thread.h>
#include <evl/clock.h>
#include <evl/sched.h>
#include <string>
#include <vector>

class TaskManager {
public:
    TaskManager(const std::string& name, int priority, int period_ns);
    void start();
    void stop();
    void executeTask();

    // Deadline Monitoring
    bool checkDeadlineMiss();

private:
    struct evl_thread thread_;
    struct timespec period_;
    int priority_;
    std::string name_;
};

#endif // TASK_MANAGER_H
