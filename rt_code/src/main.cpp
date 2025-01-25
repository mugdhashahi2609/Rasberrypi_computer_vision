#include "scheduler.h"
#include "task_manager.h"
#include <iostream>

int main() {
    try {
        // Create tasks
        TaskManager task1("Task1", 1, 100000000); // 100ms
        TaskManager task2("Task2", 2, 200000000); // 200ms

        // Initialize Scheduler
        Scheduler scheduler;
        scheduler.addTask(task1);
        scheduler.addTask(task2);

        // Run tasks
        scheduler.run();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
